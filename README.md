# ProNeuro-AD

MRI-based prognostic AI for Alzheimer's disease — research prototype.

**Goal:** from a standard structural MRI plus basic clinical data, estimate
1. **Amyloid status** (positive/negative), normally measured by PET, and
2. **Prognosis** — risk and time to conversion from MCI to Alzheimer's dementia.

> Research use only. Not a medical device. Not for clinical decisions.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 0 | Repo, environment, ADNI data access | done |
| 1 | Baseline: FreeSurfer measures (ADSP-PHC tables) → amyloid classifier + conversion survival model | done — MRI-only AUC 0.75, C-index 0.75 |
| 2 | FastSurfer on downloaded baseline T1 MRIs → normalized brain volumes | code ready |
| 3 | Deep model: tri-planar parallel ViT fine-tuning (amyloid + conversion) | code ready |
| 4 | External validation (OASIS-3, an Asian cohort), calibration, write-up | planned |

## Layout

```
configs/            YAML experiment configs
data/raw/           ADNI / OASIS downloads (never committed)
data/processed/     derived tables & images (never committed)
docs/               data access, model card, notes
scripts/            command-line entry points
src/proneuro/      the Python package
  data/             loading & label building
  preprocessing/    MRI preprocessing wrappers (FastSurfer)
  features/         feature extraction
  models/           classical & deep models, evaluation
tests/              unit tests (synthetic data only)
outputs/            results, figures (never committed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # core + tests
pip install -e ".[deep]"             # later: PyTorch/MONAI for Stage 3
pytest
```

For an RTX 50-series GPU, install a PyTorch build with CUDA 12.8 or later
(see pytorch.org for the current install command) before `.[deep]`.

## Running the Stage 1 baseline

1. Get ADNI access (see `docs/DATA_ACCESS.md`) and download `ADNIMERGE.csv`
   into `data/raw/`.
2. Run:
   ```bash
   python scripts/run_baseline.py --config configs/baseline.yaml
   ```
3. Results land in `outputs/`.

## Stage 2–3: MRI → tri-planar ViT

```bash
python scripts/select_baseline_mri.py --scan-list <ADNIMERGE_Key_MRI_*.csv>   # image IDs for LONI
python scripts/batch_fastsurfer.py --download-dir <LONI download folder>      # GPU, ~1 min/scan
python scripts/prepare_volumes.py                                             # -> data/processed/volumes
pip install -e ".[deep]"
python scripts/train_vit.py --config configs/vit.yaml --folds 1               # one fold first
python scripts/train_vit.py --config configs/vit.yaml                         # full 5-fold CV
```

The ViT has three parallel ImageNet-pretrained branches (sagittal, coronal, axial), each
fine-tuned on 2.5D slices, fused by a small transformer, with an amyloid head and a
discrete-time survival head. A FreeSurfer ridge baseline is trained on the same folds.

## Data rules

- No patient data in git. `.gitignore` blocks `data/` and imaging formats.
- Follow each dataset's Data Use Agreement (no redistribution, cite properly).
- Keep the drive encrypted on laptops.

## Renaming the project

Rename the folder and GitHub repo freely. To rename the Python package too,
change `proneuro` in `pyproject.toml`, the `src/proneuro/` folder, and imports.
