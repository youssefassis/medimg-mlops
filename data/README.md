# data/

Not tracked by git, except this file and `.gitignore`.

Download the [chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and unzip it into `raw/chest_xray/`:

```
raw/chest_xray/
  train/  NORMAL 1341 · PNEUMONIA 3875
  val/    NORMAL    8 · PNEUMONIA    8
  test/   NORMAL  234 · PNEUMONIA  390
```

`uv run python scripts/make_batches.py` splits `train/` and `val/` into 5 batches in `batches/`,
to simulate new data arriving over time. A patient is never in two batches, and patients that also
have images in `test/` are left out. `test/` is kept as is for evaluation.

## RSNA pneumonia (DICOM)

Used by `notebooks/explore_rsna.py`. Join the
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
