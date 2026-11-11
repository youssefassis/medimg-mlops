"""The datasets the pipeline can ingest. A source lists its DICOM files with their labels;
mlops.data.ingestion does the rest, the same way for every source."""

from collections.abc import Callable
from pathlib import Path

from mlops.data.ingestion import LabelledFile
from mlops.data.sources import rsna

SOURCES: dict[str, Callable[[Path], list[LabelledFile]]] = {"rsna": rsna.list_files}


def list_files(source: str, data_dir: Path) -> list[LabelledFile]:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}, expected one of {sorted(SOURCES)}")
    return SOURCES[source](data_dir)
