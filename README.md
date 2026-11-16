# ProNeuro-AD

Predicting amyloid status and progression to Alzheimer's dementia from structural brain MRI —
a head-to-head comparison of engineered FreeSurfer measures and vision transformers in ADNI.

> **Research use only.** Not a medical device. Not for clinical decisions.

## Questions

From a standard T1-weighted MRI (plus age, sex, education and APOE4 where stated), how well can we predict:

1. **Amyloid status** — positive or negative on amyloid PET (metric: ROC AUC)
2. **Progression** — time from MCI to Alzheimer's dementia within 5 years (metric: Harrell's C-index)

…and do deep models learned from the image beat conventional FreeSurfer measurements?

## Key results

Data: ADNI 1–4 via the ADSP Phenotype Harmonization Consortium tables. All results are
5-fold cross-validated; mean ± SD across folds. All models in a table share the same folds.

**FreeSurfer measures (2,916 participants with a baseline MRI)**

| Features | Model | Amyloid AUC | Conversion C-index |
|---|---|---|---|
| 192 MRI measures + demographics + APOE4 | Ridge | **0.82** ± 0.03 | **0.76** ± 0.01 |
| 192 MRI measures only | Ridge | 0.75 ± 0.02 | 0.75 ± 0.01 |
| Demographics + APOE4 (no MRI) | Logistic / Cox | 0.78 ± 0.04 | 0.64 ± 0.03 |

MRI carries most of the prognostic signal; for amyloid, APOE4 and demographics are strong on their own.

**MRI-only deep learning vs FreeSurfer (1,659 participants with processed MRI and a label)**

| Model | Amyloid AUC | Conversion C-index |
|---|---|---|
| FreeSurfer, 192 measures, ridge | **0.75** ± 0.03 | **0.75** ± 0.03 |
| FreeSurfer + frozen ViT features, ridge | 0.75 ± 0.03 | 0.75 ± 0.04 |
| Tri-planar ViT, fine-tuned | 0.72 ± 0.03 | 0.71 ± 0.03 |
| Tri-planar ViT, frozen ImageNet features | 0.68 ± 0.02 | 0.69 ± 0.03 |

At this cohort size, engineered FreeSurfer measures outperform vision transformers on
hippocampal MRI, whether frozen, fine-tuned or combined with FreeSurfer. Fine-tuning helps
(+0.04 amyloid, +0.02 conversion over frozen features) but does not close the gap.

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Repo, environment, ADNI data access | done |
| 1 | FreeSurfer baseline (ADSP-PHC tables) | done |
| 2 | FastSurfer on baseline T1 MRIs (1,941 of 2,719 selected scans) | in progress |
| 3 | Tri-planar ViT: whole brain → hippocampal crops → frozen, combined, fine-tuned | done |
| 4 | Plasma biomarkers (p-tau, NfL) + MRI; full cohort; subgroups; calibration | planned |
| 5 | External validation, preprint | planned |

## Method overview

1. **Labels.** Diagnosis, demographics and amyloid status from ADSP-PHC tables, aligned by age
   to each participant's first T1 MRI (diagnosis within ±6 months, amyloid PET within ±1 year).
   Conversion = first Alzheimer's-dementia diagnosis after a baseline MCI diagnosis, censored at
   last follow-up or 60 months.
2. **FreeSurfer baseline.** 192 ComBat-harmonized FreeSurfer measures; volumes divided by
   intracranial volume. Ridge-penalized logistic and Cox models.
3. **MRI preprocessing.** DICOM → NIfTI → FastSurfer segmentation (Docker, GPU). Multi-frame
   scans are combined by root-mean-square.
4. **Deep model.** Two 64 mm cubes centred on the left and (mirrored) right hippocampus. Each cube
   is viewed as 8 neighbouring-slice stacks per plane by three parallel ViT-Small branches
   (ImageNet-pretrained, 12 layers each), fused by a 2-layer transformer. Two heads: amyloid
   (binary) and a discrete-time survival head (5 yearly bins).
5. **Evaluation.** Stratified 5-fold cross-validation; a FreeSurfer ridge model is trained on
   exactly the same folds as every deep model.

## Repository layout

```
configs/            experiment configs (YAML)
docs/               data access notes
scripts/            command-line entry points
src/proneuro/       the Python package
  data/             label and feature building (ADSP-PHC, ADNIMERGE)
  preprocessing/    FastSurfer wrapper
  models/           classical models and cross-validation
  deep/             crops, tri-planar ViT, training
tests/              unit tests (synthetic data only)
data/, outputs/     local only — never committed
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU build first
pip install -e ".[dev]"
pip install -e ".[deep]" --no-deps && pip install timm nibabel dicom2nifti
pytest
```

Installing the CUDA build of PyTorch first stops pip from replacing it with a CPU-only build.

## Reproducing the results

1. **Data access.** Apply for ADNI at the LONI IDA (see `docs/DATA_ACCESS.md`). Export
   `ADSP_PHC_DEMODX`, `ADSP_PHC_T1_FS`, `ADSP_PHC_PET_Amyloid_Simple` and `APOERES` from the
   ADNIMERGE2 package to CSV in `data/raw/ADNI_tables/`.
2. **FreeSurfer baseline.**
   ```bash
   python scripts/run_baseline.py --config configs/baseline_adsp.yaml
   ```
3. **Select and download MRIs.**
   ```bash
   python scripts/select_baseline_mri.py --scan-list <ADNIMERGE_Key_MRI_*.csv or Tables_*.zip>
   ```
   Paste the IDs from `data/processed/mri_selection/image_ids_part*.txt` into LONI's
   Advanced Image Search and download the DICOM collection.
4. **Segment with FastSurfer** (Docker Desktop with GPU; resumes where it stopped):
   ```bash
   python scripts/batch_fastsurfer.py --download-dir <LONI folder> --workers 3 --threads 6
   ```
5. **Hippocampal crops and models.**
   ```bash
   python scripts/prepare_volumes.py --type mtl
   python scripts/train_vit.py --config configs/vit_mtl.yaml            # frozen ViT + combined models
   python scripts/train_vit.py --config configs/vit_mtl_finetune.yaml   # fine-tuned ViT
   ```
   Results are written to `outputs/` as `summary.json` and per-fold logs.

## Data and ethics

- No participant data is stored in this repository; `.gitignore` excludes `data/` and imaging formats.
- ADNI data are used under the ADNI Data Use Agreement. Publications must follow ADNI's
  acknowledgement and publication policies.
- Keep data on encrypted drives.

## Acknowledgement

Data used in this project were obtained from the Alzheimer's Disease Neuroimaging Initiative
(ADNI) database (adni.loni.usc.edu). The ADNI investigators contributed to the design and
implementation of ADNI and/or provided data but did not participate in the analysis or writing
of this work.

## Author

Min Myint Cho
