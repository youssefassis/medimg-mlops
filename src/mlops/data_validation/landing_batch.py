"""Locate a new batch of images in the landing bucket."""

from mlops import object_storage


def count_landing_files(batch_id: str) -> int:
    keys = object_storage.list_object_keys(object_storage.LANDING_BUCKET, prefix=f"{batch_id}/")
    if not keys:
        raise FileNotFoundError(
            f"No files found in s3://{object_storage.LANDING_BUCKET}/{batch_id}/"
        )
    return len(keys)
