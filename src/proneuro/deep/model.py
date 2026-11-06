"""Tri-planar parallel Vision Transformer for 3D brain MRI.

Three ViT branches (axial, coronal, sagittal) run in parallel. Each sees K
"2.5D" slices from its plane: 3 neighbouring slices stacked as RGB, so an
ImageNet-pretrained ViT can be fine-tuned directly. Each slice gives one
embedding; the 3*K embeddings (+ plane and position embeddings) go through a
small transformer that fuses information across planes, then two heads:

  amyloid_logit  -> amyloid PET positive (binary)

Two training modes (configs/vit.yaml, model.freeze_backbone):
  frozen   - the pretrained ViTs are run once per scan and their slice embeddings
             cached; only the fusion transformer and heads are trained. Data-efficient
             and fast: the right choice for a few hundred scans.
  finetune - all weights trained end to end (needs thousands of scans).
  hazard_logits  -> MCI -> AD dementia conversion, discrete-time hazard per time bin
                    (works with small batches, unlike a batch-level Cox loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES = ("sagittal", "coronal", "axial")


def make_backbone(name: str, pretrained: bool, img_size: int, **kw) -> nn.Module:
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=3,
                             img_size=img_size, **kw)


def extract_slices(vol: torch.Tensor, plane: int, k: int, span: float = 0.6) -> torch.Tensor:
    """vol (B, D, H, W) -> (B, k, 3, h, w): k positions over the central `span` of one axis,
    each a stack of 3 neighbouring slices."""
    b, n = vol.shape[0], vol.shape[plane + 1]
    centers = torch.linspace(n * (1 - span) / 2, n * (1 + span) / 2 - 1, k).round().long()
    centers = centers.clamp(1, n - 2)
    v = vol.movedim(plane + 1, 1)  # (B, n, a, c)
    idx = torch.stack([centers - 1, centers, centers + 1], dim=1).to(vol.device)  # (k, 3)
    return v[:, idx]  # (B, k, 3, a, c)


class TriPlanarViT(nn.Module):
    def __init__(self, backbone: str = "vit_small_patch16_224", pretrained: bool = True,
                 slices_per_plane: int = 8, img_size: int = 224, fusion_layers: int = 2,
                 shared_backbone: bool = False, dropout: float = 0.1, time_bins: int = 5,
                 grad_checkpointing: bool = False, regions: int = 1, **backbone_kw):
        super().__init__()
        self.k, self.img_size, self.regions = slices_per_plane, img_size, regions
        n = 1 if shared_backbone else len(PLANES)
        self.backbones = nn.ModuleList(
            make_backbone(backbone, pretrained, img_size, **backbone_kw) for _ in range(n))
        if grad_checkpointing:
            for net in self.backbones:
                net.set_grad_checkpointing(True)
        dim = self.backbones[0].num_features
        self.plane_emb = nn.Parameter(torch.zeros(len(PLANES), 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, slices_per_plane, dim))
        self.region_emb = nn.Parameter(torch.zeros(regions, 1, 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.plane_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.region_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, nhead=max(1, dim // 64), dim_feedforward=dim * 2,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, fusion_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.amyloid_head = nn.Linear(dim, 1)
        self.hazard_head = nn.Linear(dim, time_bins)

    def backbone_parameters(self):
        return self.backbones.parameters()

    @property
    def dim(self) -> int:
        return self.plane_emb.shape[-1]

    def embed(self, vol: torch.Tensor) -> torch.Tensor:
        """Run the parallel ViT branches.

        vol (B, D, H, W), or (B, R, D, H, W) for R regions (e.g. left/right hippocampus)
        -> slice embeddings (B, R*3, k, dim)."""
        if vol.ndim == 5:
            b, r = vol.shape[:2]
            emb = self.embed(vol.flatten(0, 1))  # (B*R, 3, k, dim)
            return emb.view(b, r * len(PLANES), self.k, -1)
        b = vol.shape[0]
        out = []
        for p in range(len(PLANES)):
            s = extract_slices(vol, p, self.k).flatten(0, 1).float()  # (B*k, 3, h, w)
            if s.shape[-1] != self.img_size:
                s = F.interpolate(s, size=(self.img_size,) * 2, mode="bilinear",
                                  align_corners=False)
            net = self.backbones[p % len(self.backbones)]
            out.append(net(s).view(b, self.k, -1))
        return torch.stack(out, dim=1)

    def head(self, emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fuse slice embeddings (B, R*3, k, dim) across planes and regions, and predict."""
        b, rp = emb.shape[:2]
        r = rp // len(PLANES)
        plane = self.plane_emb.repeat(r, 1, 1)                              # (R*3, 1, dim)
        region = self.region_emb[:r].repeat_interleave(len(PLANES), 0)[:, 0]  # (R*3, 1, dim)
        tokens = emb.float() + self.pos_emb[:, None] + (plane + region)[None]
        x = torch.cat([self.cls.expand(b, -1, -1), tokens.flatten(1, 2)], dim=1)
        x = self.drop(self.norm(self.fusion(x))[:, 0])
        return {"amyloid_logit": self.amyloid_head(x).squeeze(-1),
                "hazard_logits": self.hazard_head(x)}

    def head_parameters(self):
        bb = set(map(id, self.backbones.parameters()))
        return [p for p in self.parameters() if id(p) not in bb]

    def forward(self, vol: torch.Tensor) -> dict[str, torch.Tensor]:
        """vol: (B, D, H, W) normalized brain volume."""
        return self.head(self.embed(vol))


def time_bin(duration: torch.Tensor, horizon: float, n_bins: int) -> torch.Tensor:
    """Index of the time bin each duration falls in (0 .. n_bins-1)."""
    width = horizon / n_bins
    return (duration / width).ceil().long().sub(1).clamp(0, n_bins - 1)


def discrete_time_loss(hazard_logits: torch.Tensor, duration: torch.Tensor,
                       event: torch.Tensor, horizon: float) -> torch.Tensor:
    """Negative log-likelihood of a discrete-time (logistic hazard) survival model.

    For a person whose event/censoring falls in bin j: survive bins < j, and in bin j
    either have the event (event=1) or survive it (censored)."""
    n_bins = hazard_logits.shape[1]
    j = time_bin(duration, horizon, n_bins)
    bins = torch.arange(n_bins, device=hazard_logits.device)[None]
    survived = (bins < j[:, None]).float()
    at_j = (bins == j[:, None]).float()
    target = at_j * event[:, None].float()
    weight = survived + at_j  # bins after j carry no information
    loss = F.binary_cross_entropy_with_logits(hazard_logits, target, weight=weight,
                                              reduction="sum")
    return loss / max(len(duration), 1)


def risk_score(hazard_logits: torch.Tensor) -> torch.Tensor:
    """Cumulative hazard over the horizon: higher = earlier conversion expected."""
    return -F.logsigmoid(-hazard_logits).sum(dim=1)
