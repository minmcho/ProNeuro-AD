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
            v = v[::-1]  # left-right flip (axis 0 in FreeSurfer's conformed orientation)
        shift = rng.integers(-4, 5, size=3)
        v = np.roll(v, shift, axis=(0, 1, 2))
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
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    amy, risk = [], []
    for v, *_ in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = model(v.to(device, non_blocking=True))
        amy.append(torch.sigmoid(out["amyloid_logit"].float()).cpu())
        risk.append(risk_score(out["hazard_logits"].float()).cpu())
    return torch.cat(amy).numpy(), torch.cat(risk).numpy()


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


def _val_score(r: dict) -> float:
    vals = [r[k] for k in ("amyloid_auc", "conversion_cindex") if k in r]
    return float(np.mean(vals)) if vals else 0.0


def train_fold(cfg: dict, vol_dir: Path, lab: pd.DataFrame, train_ids, val_ids, test_ids,
               device: torch.device, log) -> dict:
    t = cfg["train"]
    m = cfg["model"]
    model = TriPlanarViT(backbone=m["backbone"], pretrained=m["pretrained"],
                         slices_per_plane=m["slices_per_plane"], img_size=m["img_size"],
                         fusion_layers=m["fusion_layers"], shared_backbone=m["shared_backbone"],
                         dropout=m["dropout"], time_bins=m["time_bins"],
                         grad_checkpointing=m.get("grad_checkpointing", False),
                         **m.get("backbone_kwargs", {})).to(device)
    bb = set(map(id, model.backbone_parameters()))
    head = [p for p in model.parameters() if id(p) not in bb]
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

    best, best_state, bad = -1.0, None, 0
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
        r = score(lab.loc[val_ids], *predict(model, va, device, amp))
        vs = _val_score(r)
        log(f"  epoch {epoch + 1:3d}  loss {total / max(n, 1):.4f}  val {r}  "
            f"({time.time() - t0:.0f}s)")
        if vs > best:
            best, bad = vs, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                log(f"  early stop at epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return score(lab.loc[test_ids], *predict(model, te, device, amp))


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
    results = []
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for f, (tr_i, te_i) in enumerate(skf.split(ids, lab["dx"])):
        if folds and f + 1 not in folds:
            continue
        tr_ids, te_ids = ids[tr_i], ids[te_i]
        tr_ids, va_ids = train_test_split(tr_ids, test_size=cfg["train"]["val_fraction"],
                                          random_state=seed, stratify=lab.loc[tr_ids, "dx"])
        log(f"Fold {f + 1}/{k}: train {len(tr_ids)}, val {len(va_ids)}, test {len(te_ids)}")
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

    summary = {"n_participants": int(len(ids)), "folds_run": len(results),
               "tri_planar_vit": summ("vit"), "freesurfer_ridge_same_folds": summ("ridge")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log_f.close()
    return summary
