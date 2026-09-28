"""S3-compatible object storage (RustFS locally) used for data and reports."""

import os

import boto3

LANDING_BUCKET = "landing"
PROCESSED_BUCKET = "processed"
REPORTS_BUCKET = "reports"
MLFLOW_BUCKET = "mlflow"
ALL_BUCKETS = (LANDING_BUCKET, PROCESSED_BUCKET, REPORTS_BUCKET, MLFLOW_BUCKET)


def _s3_client():
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])


def create_missing_buckets() -> None:
    client = _s3_client()
    existing = {bucket["Name"] for bucket in client.list_buckets()["Buckets"]}
    for bucket in ALL_BUCKETS:
        if bucket not in existing:
            client.create_bucket(Bucket=bucket)


def list_object_keys(bucket: str, prefix: str) -> list[str]:
    paginator = _s3_client().get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for page in pages for obj in page.get("Contents", [])]


def upload_text(bucket: str, key: str, text: str, content_type: str) -> str:
    _s3_client().put_object(
        Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type
    )
    return f"s3://{bucket}/{key}"
