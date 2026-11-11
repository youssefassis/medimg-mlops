# data/

Not tracked by git, except this file and `.gitignore`.

## Download

Join the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge),
accept its rules, download the data (3.7 GB) and unzip it into `raw/rsna/`:

```
raw/rsna/
  stage_2_train_images/            DICOM files, one per patient
  stage_2_test_images/             DICOM files without labels
  stage_2_train_labels.csv
  stage_2_detailed_class_info.csv
```

## What the scripts create

- `scripts/ingest_rsna.py` → `processed/rsna/`: the images that pass the checks, de-identified
  and saved as PNGs, with `manifest.csv` listing them and `rejected.csv` listing the others with
  the reasons.
- `scripts/make_batches.py` → `batches/`: `test.csv`, a held-out test set used to score every
  model, and `batch_1.csv` to `batch_5.csv`, released one by one to simulate new data.
