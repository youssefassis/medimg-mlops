# data/

Not tracked by git, except this file and `.gitignore`.

## RSNA pneumonia (DICOM)

The training data. Join the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge),
accept its rules, download the data (3.7 GB) and unzip it into `raw/rsna/`:

```
raw/rsna/
  stage_2_train_images/            26684 DICOM files, one per patient
  stage_2_test_images/              3000 DICOM files, without labels
  stage_2_train_labels.csv         one row per box, Target = 1 for a lung opacity
  stage_2_detailed_class_info.csv  the class of each row: Normal, Lung Opacity,
                                   or No Lung Opacity / Not Normal
```

### Ingesting RSNA

`uv run --env-file .env python scripts/ingest_rsna.py` turns a sample of the training images
(size in `config/ingestion.yaml`) into a curated dataset in `processed/rsna/`:

```
processed/rsna/
  images/        one PNG per accepted image, named by the start of its source file hash
  manifest.csv   path, label, patient_id, rsna_class, view, sex, age, pixel_spacing_mm,
                 source_sha256
  rejected.csv   source_sha256 and the reasons, for files that failed a check
```

- **Label**: `Lung Opacity` is PNEUMONIA; `Normal` and `No Lung Opacity / Not Normal` are NORMAL.
  `rsna_class` keeps the original class.
- **De-identification**: only the manifest's fields are kept from the DICOM header. `patient_id`
  is a keyed pseudonym (`DEID_SECRET` in `.env`), and ages above 89 are written as 90
  (HIPAA Safe Harbor). `source_sha256` links each row back to its raw file.
- **Checks**: header values (modality, view, sex, age, square pixel spacing) and pixels (contrast,
  share of near-black pixels). Failed files are listed in `rejected.csv`, never dropped silently.

### Test set and batches

`uv run python scripts/make_batches.py` splits `manifest.csv` by patient into `batches/`:

```
batches/
  test.csv              20% of patients, held out: never trained on, scores every model
  batch_1.csv … batch_5.csv   the rest, to simulate new data arriving over time
```

Every file keeps the manifest's columns and the same share of each label and view, so the AP
shortcut weighs the same in each split. Each training run logs its batch and the test set in
MLflow with a hash of their content, so a model always points to the exact images it saw.

## Pediatric chest X-rays (JPEG)

Kept for a later step, as data from another population. Download the
[chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and unzip it into `raw/chest_xray/`:

```
raw/chest_xray/
  train/  NORMAL 1341 · PNEUMONIA 3875
  val/    NORMAL    8 · PNEUMONIA    8
  test/   NORMAL  234 · PNEUMONIA  390
```
