# data/

Local working copy of datasets (gitignored, except this file). The pipeline itself reads from the
object store; `scripts/` moves data between the two.

Planned layout (phase 2):

- `raw/`: the dataset as downloaded (RSNA Pneumonia, DICOM)
- `batches/`: the raw data split into simulated monthly batches, ready to upload to `landing/`
