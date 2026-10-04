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
to simulate new data arriving over time. A patient is never in two batches. `test/` is kept as is
for evaluation.
