# Data access

## ADNI (primary)
- Apply at the LONI Image & Data Archive: https://ida.loni.usc.edu
- Read and follow the ADNI Data Use Agreement: no redistribution, required
  acknowledgment text in publications, and check terms for any commercial use.
- Start with the tabular file `ADNIMERGE.csv` (study data) for Stage 1.
- For Stage 2+, download baseline 3D T1-weighted MRI (MPRAGE / IR-SPGR).

## OASIS-3 (external validation)
- https://sites.wustl.edu/oasisbrains/ — separate application and terms.

## Asian cohorts (for Asia-market validation, later)
- J-ADNI (Japan) and Korean cohorts typically require a research collaborator.

## Security checklist
- [ ] Data kept only under `data/` (git-ignored)
- [ ] Laptop drive encrypted (BitLocker / FileVault / LUKS)
- [ ] No data in cloud storage without checking the DUA
- [ ] Backups encrypted
