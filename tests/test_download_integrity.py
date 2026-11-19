import json
import zipfile

import pytest

from scripts.download_twibot_shard import verify_extract, bind_identity


def fixture_zip(tmp_path):
    content = b'[{"id":"t1","author_id":"1"}]'
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("tweet.json", content)
    return archive, content


def test_same_size_corrupt_cached_raw_is_not_certified(tmp_path):
    archive, content = fixture_zip(tmp_path)
    raw = tmp_path / "tweet.json"
    raw.write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match="CRC"):
        verify_extract(archive, raw, "tweet.json", len(content))
    assert raw.read_bytes() == b"x" * len(content)


def test_verified_extraction_and_repeat_have_same_sha(tmp_path):
    archive, content = fixture_zip(tmp_path)
    raw = tmp_path / "tweet.json"
    first = verify_extract(archive, raw, "tweet.json", len(content))
    assert raw.read_bytes() == content
    assert verify_extract(archive, raw, "tweet.json", len(content)) == first


def test_changed_download_identity_quarantines_old_parts(tmp_path):
    identity = {"url": "https://example.org/data", "etag": "old", "bytes": 100}
    bind_identity(tmp_path, identity)
    part = tmp_path / "archive.part0"
    part.write_bytes(b"old-prefix")
    bind_identity(tmp_path, {**identity, "etag": "new"})
    assert not part.exists()
    assert any(p.read_bytes() == b"old-prefix" for p in tmp_path.glob("*.unverified-*"))
    assert (
        json.loads((tmp_path / "download-identity.json").read_text())["etag"] == "new"
    )


def test_unbound_parts_are_not_resumed(tmp_path):
    part = tmp_path / "archive.part0"
    part.write_bytes(b"unbound")
    bind_identity(
        tmp_path, {"url": "https://example.org/data", "etag": "current", "bytes": 100}
    )
    assert not part.exists()
