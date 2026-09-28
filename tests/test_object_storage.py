import boto3
import pytest
from botocore.stub import Stubber

from mlops import object_storage


@pytest.fixture
def s3_stub(monkeypatch):
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    monkeypatch.setattr(object_storage, "_s3_client", lambda: client)
    with Stubber(client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


def test_list_object_keys_follows_pagination(s3_stub):
    s3_stub.add_response(
        "list_objects_v2",
        {
            "Contents": [{"Key": "batch_001/a.dcm"}],
            "IsTruncated": True,
            "NextContinuationToken": "t",
        },
        {"Bucket": "landing", "Prefix": "batch_001/"},
    )
    s3_stub.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "batch_001/b.dcm"}], "IsTruncated": False},
        {"Bucket": "landing", "Prefix": "batch_001/", "ContinuationToken": "t"},
    )
    keys = object_storage.list_object_keys("landing", prefix="batch_001/")
    assert keys == ["batch_001/a.dcm", "batch_001/b.dcm"]


def test_list_object_keys_of_empty_prefix(s3_stub):
    s3_stub.add_response(
        "list_objects_v2", {"IsTruncated": False}, {"Bucket": "landing", "Prefix": "missing/"}
    )
    assert object_storage.list_object_keys("landing", prefix="missing/") == []


def test_create_missing_buckets_skips_existing(s3_stub):
    s3_stub.add_response(
        "list_buckets", {"Buckets": [{"Name": name} for name in object_storage.ALL_BUCKETS[1:]]}
    )
    s3_stub.add_response("create_bucket", {}, {"Bucket": object_storage.ALL_BUCKETS[0]})
    object_storage.create_missing_buckets()


def test_upload_text_returns_s3_uri(s3_stub):
    s3_stub.add_response(
        "put_object",
        {},
        {"Bucket": "reports", "Key": "r.html", "Body": b"<p>ok</p>", "ContentType": "text/html"},
    )
    uri = object_storage.upload_text("reports", "r.html", "<p>ok</p>", "text/html")
    assert uri == "s3://reports/r.html"
