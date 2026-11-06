"""Cross-validated fine-tuning of the tri-planar ViT, with a ridge baseline on the same folds.

The same participants and folds are used for the deep model and for the ridge
model on FreeSurfer measures, so the comparison is fair.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, Dataset

from proneuro.deep.model import TriPlanarViT, discrete_time_loss, risk_score
from proneuro.models.baseline import _classifier, _survival
from sksurv.util import Surv


class VolumeDataset(Dataset):
    """Brain volumes ({RID}.npy) with amyloid and conversion labels (NaN = missing)."""

    def __init__(self, vol_dir: Path, rids, labels: pd.DataFrame, augment: bool = False):
        self.vol_dir, self.rids, self.labels, self.augment = Path(vol_dir), list(rids), labels, augment

    def __len__(self) -> int:
        return len(self.rids)

    def _augment(self, v: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            v = np.flip(v, axis=-3)  # left-right (axis 0 of each volume, FreeSurfer LIA)
        shift = rng.integers(-4, 5, size=3)
        v = np.roll(v, shift, axis=(-3, -2, -1))
        v = v * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        return v

    def __getitem__(self, i):
        rid = self.rids[i]
        v = np.load(self.vol_dir / f"{int(rid)}.npy", mmap_mode="r").astype(np.float32)
        if self.augment:
            v = self._augment(v)
        row = self.labels.loc[rid]
        return (torch.from_numpy(np.ascontiguousarray(v)),
                torch.tensor(row["amyloid"], dtype=torch.float32),
                torch.tensor(row["duration"], dtype=torch.float32),
                torch.tensor(row["event"], dtype=torch.float32))


def build_labels(d: dict) -> pd.DataFrame:
    lab = pd.DataFrame(index=d["X"].index)
    lab["amyloid"] = d["amyloid"]
    lab = lab.join(d["conversion"][["duration", "event"]])
    lab["dx"] = d["baseline_dx"]
    return lab


def multitask_loss(out, amy, dur, evt, horizon: float, amy_weight: float = 1.0):
    loss = out["amyloid_logit"].sum() * 0.0
    m = ~torch.isnan(amy)
    if m.any():
        loss = loss + amy_weight * F.binary_cross_entropy_with_logits(
            out["amyloid_logit"][m], amy[m])
    s = ~torch.isnan(dur)
    if s.any():
        loss = loss + discrete_time_loss(out["hazard_logits"][s], dur[s], evt[s], horizon)
    return loss


@torch.no_grad()
def predict(model, loader, device, amp: bool, horizon: float = 60.0,
            amy_weight: float = 1.0, forward=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Predictions plus mean loss. `forward` maps a batch input to model outputs;
    default is model(x). For cached features with several views, outputs are averaged."""
    model.eval()
    forward = forward or model
    amy, risk, losses, n = [], [], 0.0, 0
    for x, a, d, e in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = forward(x)
        out = {k: v.float() for k, v in out.items()}
        a, d, e = a.to(device), d.to(device), e.to(device)
        losses += multitask_loss(out, a, d, e, horizon, amy_weight).item() * len(a)
        n += len(a)
        amy.append(torch.sigmoid(out["amyloid_logit"]).cpu())
        risk.append(risk_score(out["hazard_logits"]).cpu())
    return torch.cat(amy).numpy(), torch.cat(risk).numpy(), losses / max(n, 1)


def score(lab: pd.DataFrame, amy_pred: np.ndarray, risk: np.ndarray) -> dict:
    res = {}
    a = lab["amyloid"].notna().to_numpy()
    if a.sum() > 1 and lab["amyloid"][a].nunique() == 2:
        res["amyloid_auc"] = float(roc_auc_score(lab["amyloid"][a], amy_pred[a]))
    s = lab["duration"].notna().to_numpy()
    if s.sum() > 1 and lab["event"][s].sum() > 0:
        res["conversion_cindex"] = float(concordance_index_censored(
            lab["event"][s].astype(bool).to_numpy(), lab["duration"][s].to_numpy(), risk[s])[0])
    return res


def _fmt(r: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in r.items()) + "}"


def build_model(cfg: dict, pretrained: bool | None = None) -> TriPlanarViT:
    m = cfg["model"]
    return TriPlanarViT(backbone=m["backbone"],
                        pretrained=m["pretrained"] if pretrained is None else pretrained,
                        slices_per_plane=m["slices_per_plane"], img_size=m["img_size"],
                        fusion_layers=m["fusion_layers"], shared_backbone=m["shared_backbone"],
                        dropout=m["dropout"], time_bins=m["time_bins"],
                        grad_checkpointing=m.get("grad_checkpointing", False),
                        regions=m.get("regions", 1), **m.get("backbone_kwargs", {}))


# ---------------------------------------------------------------- frozen-backbone mode

@torch.no_grad()
def extract_features(cfg: dict, vol_dir: Path, rids, cache_dir: Path, device: torch.device,
                     log) -> None:
    """Run the pretrained ViT branches once per scan and cache slice embeddings.

    Saves <cache_dir>/<RID>.npy with shape (2, R*3, k, dim): view 0 = original,
    view 1 = left-right flipped (used as augmentation and test-time averaging)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rids if not (cache_dir / f"{int(r)}.npy").exists()]
    if not todo:
        log(f"Features cached for all {len(rids)} scans ({cache_dir})")
        return
    log(f"Extracting ViT features for {len(todo)} scans -> {cache_dir}")
    model = build_model(cfg).to(device).eval()
    amp = cfg["train"]["amp"] and device.type == "cuda"
    t0 = time.time()
    for i, rid in enumerate(todo):
        v = np.load(Path(vol_dir) / f"{int(rid)}.npy").astype(np.float32)
        flipped = np.ascontiguousarray(np.flip(v, axis=-3))
        x = torch.from_numpy(np.stack([v, flipped])).to(device)
        with torch.autocast(device.type, enabled=amp):
            emb = model.embed(x).float().cpu().numpy().astype(np.float16)
        np.save(cache_dir / f"{int(rid)}.npy", emb)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            log(f"  {i + 1}/{len(todo)} scans ({time.time() - t0:.0f}s)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


class FeatureDataset(Dataset):
    """Cached slice embeddings with labels. Training: a random view; eval: all views."""

    def __init__(self, cache_dir: Path, rids, labels: pd.DataFrame, train: bool = False):
        self.feats = {r: np.load(Path(cache_dir) / f"{int(r)}.npy") for r in rids}
        self.rids, self.labels, self.train = list(rids), labels, train

    def __len__(self) -> int:
        return len(self.rids)

    def __getitem__(self, i):
        rid = self.rids[i]
        f = self.feats[rid].astype(np.float32)  # (views, 3, k, dim)
        if self.train:
            f = f[np.random.randint(len(f))]
        row = self.labels.loc[rid]
        return (torch.from_numpy(f), torch.tensor(row["amyloid"], dtype=torch.float32),
                torch.tensor(row["duration"], dtype=torch.float32),
                torch.tensor(row["event"], dtype=torch.float32))


def _multi_view_head(model):
    """Average head outputs over views: x (B, views, 3, k, dim)."""
    def fwd(x):
        b, v = x.shape[:2]
        out = model.head(x.flatten(0, 1))
        return {k: o.view(b, v, *o.shape[1:]).mean(1) for k, o in out.items()}
    return fwd


def train_frozen_fold(cfg: dict, cache_dir: Path, lab: pd.DataFrame, train_ids, val_ids,
                      test_ids, device: torch.device, log) -> dict:
    fz, horizon = cfg["frozen"], cfg["horizon_months"]
    model = build_model(cfg, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.head_parameters(), lr=fz["lr"], weight_decay=fz["weight_decay"])
    tr = DataLoader(FeatureDataset(cache_dir, train_ids, lab, train=True),
                    batch_size=fz["batch_size"], shuffle=True,
                    drop_last=len(train_ids) > fz["batch_size"])
    va = DataLoader(FeatureDataset(cache_dir, val_ids, lab), batch_size=64)
    te = DataLoader(FeatureDataset(cache_dir, test_ids, lab), batch_size=64)
    fwd = _multi_view_head(model)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(fz["epochs"]):
        model.train()
        total, n = 0.0, 0
        for x, a, d, e in tr:
            x, a, d, e = (t.to(device) for t in (x, a, d, e))
            out = model.head(x)
            loss = multitask_loss(out, a, d, e, horizon, cfg["train"]["amyloid_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head_parameters(), 1.0)
            opt.step()
            total, n = total + loss.item(), n + 1
        pa, pr, vloss = predict(model, va, device, False, horizon,
                                cfg["train"]["amyloid_weight"], forward=fwd)
        if vloss < best:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if not k.startswith("backbones.")}
            if (epoch + 1) % 10 == 0 or epoch < 3:
                log(f"  epoch {epoch + 1:3d}  loss {total / max(n, 1):.4f}  "
                    f"val_loss {vloss:.4f}  val {_fmt(score(lab.loc[val_ids], pa, pr))}")
        else:
            bad += 1
            if bad >= fz["patience"]:
                log(f"  early stop at epoch {epoch + 1} (best val_loss {best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    pa, pr, _ = predict(model, te, device, False, horizon, forward=fwd)
    return score(lab.loc[test_ids], pa, pr)


def train_fold(cfg: dict, vol_dir: Path, lab: pd.DataFrame, train_ids, val_ids, test_ids,
               device: torch.device, log) -> dict:
    t = cfg["train"]
    model = build_model(cfg).to(device)
    head = model.head_parameters()
    opt = torch.optim.AdamW([{"params": model.backbone_parameters(), "lr": t["lr_backbone"]},
                             {"params": head, "lr": t["lr_head"]}], weight_decay=t["weight_decay"])
    mk = dict(batch_size=t["batch_size"], num_workers=t["num_workers"],
              pin_memory=device.type == "cuda")
    tr = DataLoader(VolumeDataset(vol_dir, train_ids, lab, augment=True), shuffle=True,
                    drop_last=len(train_ids) > t["batch_size"], **mk)
    va = DataLoader(VolumeDataset(vol_dir, val_ids, lab), **mk)
    te = DataLoader(VolumeDataset(vol_dir, test_ids, lab), **mk)

    steps = t["epochs"] * math.ceil(len(tr) / t["accumulate"])
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    horizon = cfg["horizon_months"]

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(t["epochs"]):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (v, amy, dur, evt) in enumerate(tr):
            v, amy, dur, evt = (x.to(device, non_blocking=True) for x in (v, amy, dur, evt))
            with torch.autocast(device.type, enabled=amp):
                out = model(v)
            out = {k: x.float() for k, x in out.items()}
            loss = multitask_loss(out, amy, dur, evt, horizon, t["amyloid_weight"]) / t["accumulate"]
            scaler.scale(loss).backward()
            if (i + 1) % t["accumulate"] == 0 or i + 1 == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            total, n = total + loss.item() * t["accumulate"], n + 1
        pa, pr, vloss = predict(model, va, device, amp, horizon, t["amyloid_weight"])
        r = score(lab.loc[val_ids], pa, pr)
        log(f"  epoch {epoch + 1:3d}  loss {total / max(n, 1):.4f}  val_loss {vloss:.4f}  "
            f"val {_fmt(r)}  ({time.time() - t0:.0f}s)")
        if vloss < best:
            best, bad = vloss, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                log(f"  early stop at epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pa, pr, _ = predict(model, te, device, amp, horizon)
    return score(lab.loc[test_ids], pa, pr)


def ridge_fold(regions: pd.DataFrame, lab: pd.DataFrame, train_ids, test_ids, seed: int) -> dict:
    """FreeSurfer all-regions ridge baseline on exactly the same fold."""
    res = {}
    a_tr = [r for r in train_ids if not np.isnan(lab.at[r, "amyloid"])]
    if a_tr:
        clf = _classifier("ridge", seed).fit(regions.loc[a_tr], lab.loc[a_tr, "amyloid"].astype(int))
        pred = clf.predict_proba(regions.loc[test_ids])[:, 1]
    else:
        pred = np.zeros(len(test_ids))
    s_tr = [r for r in train_ids if not np.isnan(lab.at[r, "duration"])]
    if s_tr:
        sv = _survival("ridge", seed).fit(
            regions.loc[s_tr], Surv.from_arrays(lab.loc[s_tr, "event"].astype(bool),
                                                lab.loc[s_tr, "duration"]))
        risk = sv.predict(regions.loc[test_ids])
    else:
        risk = np.zeros(len(test_ids))
    res.update(score(lab.loc[test_ids], pred, risk))
    return res


def cross_validate(cfg: dict, d: dict, vol_dir: Path, out_dir: Path, device: torch.device,
                   folds: list[int] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    lab = build_labels(d)
    have = {int(p.stem) for p in Path(vol_dir).glob("*.npy")}
    lab = lab[lab.index.isin(have) & lab["dx"].isin([1, 2, 3])]
    lab = lab[lab["amyloid"].notna() | lab["duration"].notna()]
    regions = d["regions"].loc[lab.index]
    ids = np.array(lab.index)
    log(f"{len(ids)} participants with a volume and a label "
        f"(amyloid {int(lab['amyloid'].notna().sum())}, conversion {int(lab['duration'].notna().sum())})")

    seed, k = cfg["seed"], cfg["n_splits"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    frozen = cfg["model"].get("freeze_backbone", False)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "no GPU"
    log(f"Device: {device} ({gpu}); mode: {'frozen backbone' if frozen else 'full fine-tuning'}")
    if frozen:
        m = cfg["model"]
        tag = (f"{Path(vol_dir).name}_{m['backbone']}_k{m['slices_per_plane']}"
               f"_s{m['img_size']}")
        cache_dir = Path(cfg["data"].get("feature_dir", "data/processed/vit_features")) / tag
        extract_features(cfg, vol_dir, ids, cache_dir, device, log)
    results = []
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for f, (tr_i, te_i) in enumerate(skf.split(ids, lab["dx"])):
        if folds and f + 1 not in folds:
            continue
        tr_ids, te_ids = ids[tr_i], ids[te_i]
        tr_ids, va_ids = train_test_split(tr_ids, test_size=cfg["train"]["val_fraction"],
                                          random_state=seed, stratify=lab.loc[tr_ids, "dx"])
        log(f"Fold {f + 1}/{k}: train {len(tr_ids)}, val {len(va_ids)}, test {len(te_ids)}")
        if frozen:
            vit = train_frozen_fold(cfg, cache_dir, lab, tr_ids, va_ids, te_ids, device, log)
        else:
            vit = train_fold(cfg, vol_dir, lab, tr_ids, va_ids, te_ids, device, log)
        ridge = ridge_fold(regions, lab, np.concatenate([tr_ids, va_ids]), te_ids, seed)
        log(f"Fold {f + 1} test  ViT {vit}  |  FreeSurfer ridge {ridge}")
        results.append({"fold": f + 1, "vit": vit, "ridge": ridge})
        (out_dir / "folds.json").write_text(json.dumps(results, indent=2))

    def summ(key: str) -> dict:
        out = {}
        for metric in ("amyloid_auc", "conversion_cindex"):
            v = [r[key][metric] for r in results if metric in r[key]]
            if v:
                out[metric] = {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)}
        return out

    summary = {"mode": "frozen" if frozen else "finetune",
               "n_participants": int(len(ids)), "folds_run": len(results),
               "tri_planar_vit": summ("vit"), "freesurfer_ridge_same_folds": summ("ridge")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log_f.close()
    return summary
