import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pydicom
    from matplotlib.patches import Rectangle
    from PIL import Image, ImageOps

    from mlops.data.dicom import to_uint8

    return Image, ImageOps, Rectangle, io, mo, np, pd, plt, pydicom, to_uint8


@app.cell
def _(mo):
    mo.md(r"""
    # Exploring the RSNA pneumonia dataset

    The [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
    (2018) released chest X-rays from the NIH ChestX-ray14 dataset, relabelled by radiologists
    who drew a box around each lung opacity.

    This notebook looks at what is really in the files before deciding how to preprocess them:
    the DICOM headers, the labels, possible shortcuts a model could learn, the pixels, and how
    the unlabelled test set compares.

    Download the data into `data/raw/rsna/` first (see `data/README.md`).
    """)
    return


@app.cell
def _(mo):
    RAW_DIR = mo.notebook_dir().parent / "data" / "raw" / "rsna"
    train_files = sorted((RAW_DIR / "stage_2_train_images").glob("*.dcm"))
    test_files = sorted((RAW_DIR / "stage_2_test_images").glob("*.dcm"))
    if not train_files:
        raise FileNotFoundError(f"No DICOM files in {RAW_DIR / 'stage_2_train_images'}")

    mo.md(f"""
    | Folder | Files |
    |---|---|
    | `stage_2_train_images/` | {len(train_files):,} |
    | `stage_2_test_images/` | {len(test_files):,} (no labels: they were the competition's hidden test set) |
    """)
    return RAW_DIR, test_files, train_files


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. One DICOM file

    A DICOM file is a header (tags such as patient, study, image settings) followed by the
    pixel data. Here is the full header of the first training image.
    """)
    return


@app.cell
def _(mo, pydicom, train_files):
    _example = pydicom.dcmread(train_files[0])
    mo.plain_text(str(_example))
    return


@app.cell
def _(mo):
    mo.md(r"""
    What stands out:

    - **SOP Class `Secondary Capture`, Conversion Type `WSD`**: the image was converted on a
      workstation, not produced by the X-ray machine. The original scanner values are gone.
    - **Transfer Syntax `JPEG Baseline`, Lossy Image Compression `01`**: the pixels were already
      compressed with loss. Saving them as JPEG again would lose more detail.
    - **Patient's Name = Patient ID = file name**: identity was already replaced by a random ID.
    - **Study Date `19010101`**: a placeholder date, so the dates can't be used.
    - **Patient's Age `51`**: DICOM expects `051Y`, so a strict parser would reject it.
    - **No Rescale Slope/Intercept, no Window Center/Width**: the stored values are shown as is.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Headers across all files

    Reading only the headers (not the pixels) of every training file takes a few seconds.
    """)
    return


@app.cell
def _(pydicom):
    def read_header(path):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        row_spacing, column_spacing = (float(value) for value in dataset.PixelSpacing)
        return {
            "patientId": path.stem,
            "transfer_syntax": dataset.file_meta.TransferSyntaxUID.name,
            "sop_class": dataset.SOPClassUID.name,
            "modality": dataset.Modality,
            "photometric": dataset.PhotometricInterpretation,
            "bits_stored": dataset.BitsStored,
            "size": f"{dataset.Rows}×{dataset.Columns}",
            "frames": int(dataset.get("NumberOfFrames", 1)),
            "slice_tags": any(
                keyword in dataset
                for keyword in ("SliceThickness", "ImagePositionPatient", "SpacingBetweenSlices")
            ),
            "square_pixels": row_spacing == column_spacing,
            "study_date": dataset.StudyDate,
            "view": dataset.ViewPosition,
            "sex": dataset.PatientSex,
            "age": int(dataset.PatientAge),
            "pixel_spacing_mm": round(row_spacing, 3),
        }

    return (read_header,)


@app.cell
def _(pd, read_header, train_files):
    headers = pd.DataFrame([read_header(path) for path in train_files])
    headers.drop(columns="patientId").agg(["nunique", lambda column: column.mode()[0]]).T.set_axis(
        ["distinct values", "most common"], axis=1
    )
    return (headers,)


@app.cell
def _(mo):
    mo.md(r"""
    Most fields have a single value: every image is a 1024×1024, 8-bit, JPEG-compressed
    `MONOCHROME2` secondary capture. The fields that vary are the ones to look at.

    ### 2D, not 3D

    Every file holds one frame and has no slice tags (`SliceThickness`, `ImagePositionPatient`):
    each file is a single 2D projection of the whole chest.

    A CT or MRI scan is different: a series of 2D slices, usually one file each, that must be
    sorted and stacked using their position and orientation tags. The spacing is often
    different along each axis (for example 0.7 × 0.7 mm within a slice but 5 mm between
    slices), so volumes are resampled to a common grid. None of that applies here.

    ### Pixel spacing

    Each image has one spacing: its pixels are square (row and column spacing are equal in every
    file). But the spacing differs between images, so the same 1024 pixels cover a different
    width of chest.

    On a radiograph the spacing is only approximate. X-rays fan out from a point source, so
    anatomy is magnified on the detector, more so the farther it is from it: the heart looks
    bigger on AP images. DICOM separates `ImagerPixelSpacing` (at the detector) from
    `PixelSpacing` (corrected to the patient). These files only have `PixelSpacing`, rescaled when
    the images were resized to 1024×1024. It is a rough scale, not a ruler.
    """)
    return


@app.cell
def _(headers, plt):
    _spacing = headers["pixel_spacing_mm"].value_counts().sort_index()
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.bar([f"{value:.3f}" for value in _spacing.index], _spacing.values)
    _ax.set_xlabel("pixel spacing (mm)")
    _ax.set_ylabel("images (log scale)")
    _ax.set_yscale("log")
    _ax.tick_params(axis="x", rotation=60)
    _secondary = _ax.secondary_xaxis("top")
    _secondary.set_xticks(range(len(_spacing)), [f"{value * 1024:.0f}" for value in _spacing.index])
    _secondary.set_xlabel("image width on the patient (mm)")
    _secondary.tick_params(axis="x", rotation=60)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(headers, mo, plt):
    _outliers = headers[headers["age"] > 100]
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.hist(headers["age"], bins=range(0, 160, 2))
    _ax.set_xlabel("age (years)")
    _ax.set_ylabel("images")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"### Age\n\n{len(_outliers)} patients are older than 100, "
                f"up to {headers['age'].max()}: data entry errors that a pipeline must catch."
            ),
            _fig,
            _outliers[["patientId", "age", "sex", "view"]],
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Labels

    Two CSV files hold the labels:

    - `stage_2_train_labels.csv`: one row per box, with `Target` = 1 for a lung opacity.
      Patients without an opacity have one row with empty coordinates.
    - `stage_2_detailed_class_info.csv`: the same rows, with one of three classes.
    """)
    return


@app.cell
def _(RAW_DIR, headers, mo, pd):
    labels = pd.read_csv(RAW_DIR / "stage_2_train_labels.csv")
    _classes = pd.read_csv(RAW_DIR / "stage_2_detailed_class_info.csv").drop_duplicates()
    patients = _classes.merge(headers, on="patientId", validate="one_to_one")
    boxes = labels.dropna(subset=["x"])
    _boxes_per_patient = boxes.groupby("patientId").size().value_counts().sort_index()

    mo.vstack(
        [
            mo.md(
                f"{len(labels):,} label rows for {labels['patientId'].nunique():,} patients, "
                f"with {len(boxes):,} boxes."
            ),
            mo.hstack(
                [
                    patients["class"].value_counts().rename("patients"),
                    _boxes_per_patient.rename_axis("boxes").rename("patients"),
                    pd.crosstab(
                        labels.drop_duplicates("patientId").set_index("patientId")["Target"],
                        _classes.set_index("patientId")["class"],
                    ),
                ],
                justify="start",
                gap=2,
            ),
        ]
    )
    return boxes, patients


@app.cell
def _(mo):
    mo.md(r"""
    `Target` = 1 is exactly the `Lung Opacity` class. `Target` = 0 mixes two classes:
    `Normal` and `No Lung Opacity / Not Normal` (something else is abnormal). How to map these
    three classes to a binary label is a decision to make before training on this data.

    ### Where the boxes are
    """)
    return


@app.cell
def _(boxes, np, plt):
    _heatmap = np.zeros((1024, 1024))
    for _box in boxes.itertuples():
        _top, _left = int(_box.y), int(_box.x)
        _heatmap[_top : _top + int(_box.height), _left : _left + int(_box.width)] += 1

    _fig, (_ax_map, _ax_size) = plt.subplots(1, 2, figsize=(10, 4))
    _ax_map.imshow(_heatmap, cmap="magma")
    _ax_map.set_title("boxes covering each pixel")
    _ax_map.axis("off")
    _ax_size.scatter(boxes["width"], boxes["height"], s=2, alpha=0.3)
    _ax_size.set_xlabel("box width (pixels)")
    _ax_size.set_ylabel("box height (pixels)")
    _ax_size.set_title("box sizes")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The boxes sit on the two lungs, mostly in the lower half. Patients with two boxes almost
    always have one per lung.

    ### Box sizes in mm, and after resizing

    With the pixel spacing, each box gets an approximate size in mm. The model will see the
    image at 224×224, and ResNet18 ends on a 7×7 grid: each cell of that grid covers
    32×32 pixels of its input.
    """)
    return


@app.cell
def _(boxes, mo, np, patients, plt):
    _sized = boxes.merge(patients[["patientId", "pixel_spacing_mm"]], on="patientId")
    _width_mm = _sized["width"] * _sized["pixel_spacing_mm"]
    _height_mm = _sized["height"] * _sized["pixel_spacing_mm"]
    _short_side_224 = np.minimum(_sized["width"], _sized["height"]) * 224 / 1024
    _below_cell = (_short_side_224 < 32).mean()

    _fig, (_ax_mm, _ax_224) = plt.subplots(1, 2, figsize=(11, 3.5))
    _ax_mm.hist([_width_mm, _height_mm], bins=40, label=["width", "height"])
    _ax_mm.set_xlabel("box size (mm)")
    _ax_mm.set_ylabel("boxes")
    _ax_mm.legend()
    _ax_224.hist(_short_side_224, bins=40)
    _ax_224.axvline(32, color="red", linestyle="--", label="one 7×7 grid cell")
    _ax_224.set_xlabel("shortest side at 224×224 (pixels)")
    _ax_224.set_ylabel("boxes")
    _ax_224.legend()
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"Median box: {_width_mm.median():.0f} × {_height_mm.median():.0f} mm. "
                f"At 224×224, {_below_cell:.1%} of boxes have a side shorter than one grid cell: "
                "opacities are large, so 224×224 keeps most of them, but the smallest ones are "
                "the ones a model is most likely to miss."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Possible shortcuts

    A model can learn anything that predicts the label, not only the disease. AP images
    (front to back) are taken with a mobile machine, often at the bedside of patients too sick
    to stand. PA images (back to front) are taken standing, in the radiology department.
    """)
    return


@app.cell
def _(mo, patients, pd, plt):
    _by_view = pd.crosstab(patients["view"], patients["class"], normalize="index")
    _by_sex = pd.crosstab(patients["sex"], patients["class"], normalize="index")

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.5))
    _by_view.plot.barh(stacked=True, ax=_axes[0], legend=False, title="class by view")
    _by_sex.plot.barh(stacked=True, ax=_axes[1], legend=False, title="class by sex")
    patients.boxplot(column="age", by="class", ax=_axes[2], vert=False)
    _axes[2].set_title("age by class")
    _axes[2].set_xlim(0, 100)
    _fig.suptitle("")
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="lower center", ncol=3)
    _fig.tight_layout(rect=(0, 0.1, 1, 1))

    _opacity = _by_view["Lung Opacity"]
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**{_opacity['AP']:.0%} of AP images have a lung opacity, against "
                f"{_opacity['PA']:.0%} of PA images.** A model can score well by recognizing "
                "the view (the image looks different: wider heart, markers, cables) instead of "
                "the opacity. Metrics must be checked per view to catch this."
            ),
        ]
    )
    return


@app.cell
def _(mo, patients, pd):
    _by_spacing = pd.crosstab(patients["pixel_spacing_mm"], patients["view"])
    _by_spacing["Lung Opacity (%)"] = (
        pd.crosstab(patients["pixel_spacing_mm"], patients["class"], normalize="index")[
            "Lung Opacity"
        ]
        * 100
    ).round(0)

    mo.vstack(
        [
            mo.md(r"""
    ### Pixel spacing gives away the view

    Each spacing value most likely comes from one kind of machine: a fixed stand in the
    radiology room or a mobile unit. Some values are almost only PA, others mostly AP, and the
    share of opacities follows. The model never reads the spacing tag, but it sees what the
    spacing causes: how much of the chest fills the image.
    """),
            _by_spacing,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A sample of decoded images

    The next views need the pixels. Decoding is much slower than reading headers, so they share
    one random sample per class and view, decoded once (about 30 seconds). Each image is kept
    at 256×256, with its top strip at higher resolution and its full-resolution statistics.
    """)
    return


@app.cell
def _(Image, RAW_DIR, np, pd, pydicom):
    def read_image(patient_id):
        return pydicom.dcmread(RAW_DIR / "stage_2_train_images" / f"{patient_id}.dcm")

    def summarize_pixels(pixels):
        """Statistics at full resolution, plus small copies to display."""
        return {
            "small": np.asarray(Image.fromarray(pixels).resize((256, 256))),
            "top_strip": np.asarray(Image.fromarray(pixels[:192]).resize((512, 96))),
            "histogram": np.bincount(pixels.ravel(), minlength=256),
            "min": int(pixels.min()),
            "max": int(pixels.max()),
            "mean": float(pixels.mean()),
            "std": float(pixels.std()),
            "near_black": float((pixels < 10).mean()),
        }

    def decode_sample(patients, per_group, seed):
        sample = patients.groupby(["class", "view"]).sample(per_group, random_state=seed)
        rows = [
            {"patientId": patient_id, "class": label, "view": view}
            | summarize_pixels(read_image(patient_id).pixel_array)
            for patient_id, label, view in sample[["patientId", "class", "view"]].itertuples(
                index=False
            )
        ]
        return pd.DataFrame(rows)

    return decode_sample, read_image


@app.cell
def _(decode_sample, patients):
    pixel_sample = decode_sample(patients, per_group=80, seed=0)
    pixel_sample.groupby(["view", "class"]).size().unstack()
    return (pixel_sample,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Average image per class and view

    Averaging many images blurs out each patient and keeps what they share: positioning, field
    of view, and anything always at the same place, like markers.
    """)
    return


@app.cell
def _(np, pixel_sample, plt):
    _classes = sorted(pixel_sample["class"].unique())
    _fig, _axes = plt.subplots(2, len(_classes), figsize=(12, 8))
    for _row, _view in enumerate(["AP", "PA"]):
        for _column, _label in enumerate(_classes):
            _group = pixel_sample[
                (pixel_sample["view"] == _view) & (pixel_sample["class"] == _label)
            ]
            _ax = _axes[_row, _column]
            _ax.imshow(np.mean(np.stack(_group["small"]), axis=0), cmap="gray", vmin=0, vmax=255)
            _ax.set_title(f"{_view} · {_label}\n({len(_group)} images)", fontsize=9)
            _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    AP and PA averages differ more than the classes within one view. PA averages show dark,
    sharp lung fields and a clear spine: patients stand in the same position every time. AP
    averages are blurrier and flatter, with a wider heart shadow: bedside patients are
    positioned less consistently. Within each view, the `Lung Opacity` average has hazier,
    lighter lungs. That is the real signal, and it is weaker than the AP/PA difference.

    ### Burned-in text and markers

    Text is often written into the pixels themselves: `PORTABLE`, the view, L/R markers. A model
    can read it. Here are the top strips of random images from the sample.
    """)
    return


@app.cell
def _(pixel_sample, plt):
    _strips_per_view = 10
    _fig, _axes = plt.subplots(_strips_per_view, 2, figsize=(12, 10))
    for _column, _view in enumerate(["AP", "PA"]):
        _strips = pixel_sample[pixel_sample["view"] == _view].sample(
            _strips_per_view, random_state=1
        )["top_strip"]
        for _row, _strip in enumerate(_strips):
            _axes[_row, _column].imshow(_strip, cmap="gray", vmin=0, vmax=255)
            _axes[_row, _column].axis("off")
        _axes[0, _column].set_title(_view)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    L/R markers appear in both views, but words like `PORTABLE` and `AP` only on AP images,
    together with ECG electrodes and cables. Real pipelines mask these areas, crop to the
    lungs, or at least check whether the model looks at them (for example with saliency maps).

    ## 5. Images

    Pick a class and a view to see random images, with the label boxes in red.
    """)
    return


@app.cell
def _(mo, patients):
    class_picker = mo.ui.dropdown(
        options=sorted(patients["class"].unique()), value="Lung Opacity", label="Class"
    )
    view_picker = mo.ui.dropdown(options=["AP", "PA"], value="AP", label="View")
    seed_picker = mo.ui.number(start=0, stop=10_000, value=0, label="Random sample")
    mo.hstack([class_picker, view_picker, seed_picker], justify="start", gap=2)
    return class_picker, seed_picker, view_picker


@app.cell
def _(
    Rectangle,
    boxes,
    class_picker,
    patients,
    plt,
    read_image,
    seed_picker,
    view_picker,
):
    _selected = patients[
        (patients["class"] == class_picker.value) & (patients["view"] == view_picker.value)
    ]
    shown_ids = _selected.sample(8, random_state=seed_picker.value)["patientId"].tolist()

    _fig, _axes = plt.subplots(2, 4, figsize=(13, 7))
    for _patient_id, _ax in zip(shown_ids, _axes.flat, strict=True):
        _ax.imshow(read_image(_patient_id).pixel_array, cmap="gray", vmin=0, vmax=255)
        for _box in boxes[boxes["patientId"] == _patient_id].itertuples():
            _ax.add_patch(
                Rectangle((_box.x, _box.y), _box.width, _box.height, fill=False, color="red")
            )
        _ax.set_title(_patient_id[:8], fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return (shown_ids,)


@app.cell
def _(mo):
    mo.md(r"""
    Things to look for: burned-in text and L/R markers, cables and electrodes (common on AP),
    rotated or cropped chests, and very different brightness and contrast between images.

    ## 6. Pixel intensities
    """)
    return


@app.cell
def _(np, pixel_sample, plt):
    _fig, (_ax_hist, _ax_range) = plt.subplots(1, 2, figsize=(12, 3.5))
    for _view, _group in pixel_sample.groupby("view"):
        _mean = np.mean(np.stack(_group["histogram"]), axis=0)
        _ax_hist.plot(_mean / _mean.sum(), label=_view)
        _ax_range.scatter(_group["min"], _group["max"], s=8, alpha=0.5, label=_view)
    _ax_hist.set_xlabel("stored pixel value")
    _ax_hist.set_ylabel("share of pixels")
    _ax_hist.set_title("average histogram")
    _ax_hist.legend()
    _ax_range.set_xlabel("darkest pixel")
    _ax_range.set_ylabel("brightest pixel")
    _ax_range.set_title("range used by each image")
    _ax_range.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Images don't all use the full 0–255 range, and AP and PA images have different histograms:
    another way a model can tell the views apart. Stretching each image to its own range
    (what `to_uint8` does) removes part of that difference.

    ### Image quality outliers

    The extremes of the sample, by brightness, contrast and share of near-black pixels. These
    are where rotated, inverted, cropped or unusable images show up.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The darkest images are mostly not dark at all: the X-ray fills only part of the frame,
    surrounded by black borders, and is sometimes tilted. A model resizing the whole frame sees
    a smaller chest. The brightest and lowest-contrast images are washed out but readable.
    Children appear too: a different anatomy, often with different devices.
    """)
    return


@app.cell
def _(pixel_sample, plt):
    _rankings = [
        ("darkest", "mean", True),
        ("brightest", "mean", False),
        ("lowest contrast", "std", True),
        ("most near-black pixels", "near_black", False),
    ]
    _per_ranking = 5
    _fig, _axes = plt.subplots(len(_rankings), _per_ranking, figsize=(12, 10))
    for _row, (_title, _column, _ascending) in enumerate(_rankings):
        _extremes = pixel_sample.sort_values(_column, ascending=_ascending).head(_per_ranking)
        for _ax, _image in zip(_axes[_row], _extremes.itertuples(), strict=True):
            _ax.imshow(_image.small, cmap="gray", vmin=0, vmax=255)
            _ax.set_title(f"{_image.view} · {_column} {getattr(_image, _column):.2f}", fontsize=8)
            _ax.axis("off")
        _axes[_row, 0].text(
            -0.1,
            0.5,
            _title,
            transform=_axes[_row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
        )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Preprocessing choices

    The first image of the grid in section 5, through the options to compare.
    """)
    return


@app.cell
def _(Image, ImageOps, io, mo, np, plt, read_image, shown_ids, to_uint8):
    _dataset = read_image(shown_ids[0])
    _stored = _dataset.pixel_array
    _stretched = to_uint8(_dataset)
    _equalized = np.asarray(ImageOps.equalize(Image.fromarray(_stretched)))
    _resized = Image.fromarray(_stretched).resize((224, 224), Image.Resampling.BILINEAR)
    _resized_back = np.asarray(_resized.resize((1024, 1024), Image.Resampling.BILINEAR))

    def _round_trip(image, image_format):
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format=image_format)
        return np.asarray(Image.open(buffer))

    _png_error = np.abs(_round_trip(_stretched, "PNG").astype(int) - _stretched)
    _jpeg_error = np.abs(_round_trip(_stretched, "JPEG").astype(int) - _stretched)
    _stretch_note = (
        "this image already uses the full range, so stretching changes nothing here; "
        "try other samples"
        if (_stored.min(), _stored.max()) == (0, 255)
        else "stretched they span 0–255"
    )

    _crop = np.s_[300:556, 200:456]
    _panels = [
        ("stored pixels", _stored, {"vmin": 0, "vmax": 255}),
        ("stretched to 0–255 (to_uint8)", _stretched, {}),
        ("histogram equalized", _equalized, {}),
        ("crop at 1024×1024", _stretched[_crop], {}),
        ("same crop after 224×224", _resized_back[_crop], {}),
        ("JPEG error ×10", np.clip(_jpeg_error * 10, 0, 255), {"vmin": 0, "vmax": 255}),
    ]
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 8))
    for (_title, _image, _options), _ax in zip(_panels, _axes.flat, strict=True):
        _ax.imshow(_image, cmap="gray", **_options)
        _ax.set_title(_title, fontsize=10)
        _ax.axis("off")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(f"""
    - **Stretching**: stored values span {_stored.min()}–{_stored.max()};
      {_stretch_note}.
    - **Histogram equalization** boosts contrast in the lungs, but also boosts noise and
      changes the image more than a radiologist's window would.
    - **Resizing to 224×224**, the input size of ResNet18, divides the pixel count by 21.
      Small opacities can disappear.
    - **Saving**: PNG is lossless (largest error: {_png_error.max()}). JPEG adds error on top
      of the original compression (mean {_jpeg_error.mean():.2f}, largest {_jpeg_error.max()}).
    """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Train vs test

    The test set has no labels, but its headers can still be compared with the training set.
    If they differ, metrics measured on a split of the training data won't predict
    performance on the test set: a distribution shift.
    """)
    return


@app.cell
def _(headers, mo, pd, plt, read_header, test_files):
    test_headers = pd.DataFrame([read_header(path) for path in test_files])
    _both = pd.concat(
        [headers.assign(split="train"), test_headers.assign(split="test")], ignore_index=True
    )

    def _shares(column):
        return (pd.crosstab(_both[column], _both["split"], normalize="columns") * 100).round(1)

    _fig, _ax = plt.subplots(figsize=(9, 3))
    for _split, _group in _both.groupby("split"):
        _ax.hist(_group["age"], bins=range(0, 100, 2), density=True, histtype="step", label=_split)
    _ax.set_xlabel("age (years)")
    _ax.set_ylabel("share of images")
    _ax.legend()
    _fig.tight_layout()

    _ap_gap = abs(_shares("view").loc["AP", "train"] - _shares("view").loc["AP", "test"])
    mo.vstack(
        [
            mo.md("Share of images (%) in each split:"),
            mo.hstack(
                [_shares("view"), _shares("sex"), _shares("pixel_spacing_mm")],
                justify="start",
                gap=2,
            ),
            _fig,
            mo.md(
                f"The share of AP images differs by {_ap_gap:.1f} points between train and test. "
                f"Test ages above 100: {(test_headers['age'] > 100).sum()}."
            ),
        ]
    )
    return (test_headers,)


@app.cell
def _(headers, mo, patients, test_headers):
    _ap_share = (headers["view"] == "AP").mean()
    _test_ap_share = (test_headers["view"] == "AP").mean()
    mo.md(f"""
    ## 9. What this means for the pipeline

    | Challenge | What we found | What to do |
    |---|---|---|
    | Data lineage | Secondary captures of NIH PNGs, lossy JPEG, 8-bit | Record the source; never re-save as JPEG |
    | Geometry | 2D projections; {headers["pixel_spacing_mm"].nunique()} pixel spacings at the same 1024 size | Keep spacing in the manifest; treat it as approximate |
    | Invalid values | {(headers["age"] > 100).sum()} ages above 100, age not in DICOM format | Validate header values at ingestion |
    | Labels | 3 classes, `Target` merges two of them | Choose the binary mapping on purpose |
    | View shortcut | {_ap_share:.0%} of images are AP, and opacities are much more common on AP | Report metrics per view |
    | Machine shortcut | Pixel spacing almost tells the view, and the opacity rate with it | Report metrics per spacing group; never feed spacing to the model |
    | Burned-in text | L/R markers in both views; `PORTABLE` and `AP` text only on AP | Mask or crop, and check where the model looks |
    | Image quality | Tilted images, chests framed by black borders, washed-out images | Add quality checks at ingestion |
    | Train/test shift | {_ap_share:.0%} AP in train vs {_test_ap_share:.0%} in test | Compare distributions before trusting a metric |
    | Class balance | {(patients["class"] == "Lung Opacity").mean():.0%} have an opacity | Keep the balance in each split |
    | Hidden duplicates | One image per random ID, but the NIH source has several per patient | Look for near-duplicates before splitting |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
