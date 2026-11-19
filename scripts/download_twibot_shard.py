"""Download a public third-party tweet shard; never claim it is full TwiBot-22.

Archive ranges resume independently. Extraction is restricted to the one verified
filename, with size and ZIP CRC checks. No remote code or pickle is executed.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import uuid
import zlib

import httpx

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/abdelwahedomar/twibot"
FILE = "tweet_7-004.json"
EXPECTED_RAW_BYTES = 11923856722
DEST = ROOT / "data/raw/twibot22-kaggle-shard7"


def bind_identity(directory, identity):
    """Never join byte prefixes from two archive versions; preserve old caches."""
    sidecar = directory / "download-identity.json"
    old = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if old != identity:
        tag = uuid.uuid4().hex[:12]
        cached = [
            p
            for p in directory.glob("archive.part*")
            if re.fullmatch(r"archive\.part\d+", p.name)
        ]
        cached += [
            directory / name
            for name in ("twibot-shard7.zip", FILE, "download-manifest.json")
        ]
        for path in cached:
            if path.exists():
                path.rename(path.with_name(path.name + f".unverified-{tag}"))
    sidecar.write_text(json.dumps(identity, indent=2) + "\n")


def verify_extract(archive, output, name, expected_bytes):
    """Verify both fresh and already existing raw files against ZIP CRC metadata."""
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != [name]:
            raise ValueError("Unexpected archive members")
        info = zipped.getinfo(name)
        if info.file_size != expected_bytes:
            raise ValueError("Unexpected uncompressed size")
        existing = output.exists()
        if existing and output.stat().st_size != expected_bytes:
            raise ValueError("Cached raw size mismatch")
        temporary = output.with_suffix(output.suffix + ".partial")
        source = output.open("rb") if existing else zipped.open(info)
        target = None if existing else temporary.open("wb")
        copied = 0
        crc = 0
        sha = hashlib.sha256()
        try:
            with source:
                while chunk := source.read(4 * 1024 * 1024):
                    copied += len(chunk)
                    if copied > expected_bytes:
                        raise ValueError("Extraction size limit exceeded")
                    sha.update(chunk)
                    crc = zlib.crc32(chunk, crc)
                    if target is not None:
                        target.write(chunk)
        finally:
            if target is not None:
                target.close()
        if copied != expected_bytes:
            raise ValueError("Raw size mismatch")
        if crc & 0xFFFFFFFF != info.CRC:
            raise ValueError("Raw CRC mismatch")
        if not existing:
            temporary.rename(output)
        return sha.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "twibot-shard7.zip"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with client.stream("GET", URL, headers={"Range": "bytes=0-0"}) as response:
            response.raise_for_status()
            match = re.fullmatch(
                r"bytes 0-0/(\d+)", response.headers.get("Content-Range", "")
            )
            if response.status_code != 206 or not match:
                raise RuntimeError("Server does not support verified ranges")
            total = int(match[1])
            etag = response.headers.get("etag")
            google_hash = response.headers.get("x-goog-hash", "")
            md5_match = re.search(r"(?:^|,)\s*md5=([^,]+)", google_hash)
            expected_md5 = base64.b64decode(md5_match[1]).hex() if md5_match else None
    print(
        f"Public archive size {total} bytes, expected raw file {EXPECTED_RAW_BYTES} bytes",
        flush=True,
    )
    if shutil.disk_usage(DEST).free < total + EXPECTED_RAW_BYTES + 2 * 1024**3:
        raise RuntimeError("Insufficient free disk for archive and extraction")
    identity = {"url": URL, "etag": etag, "bytes": total}
    # A pre-sidecar cache can only be adopted by matching the server content hash.
    sidecar = DEST / "download-identity.json"
    if (
        not sidecar.exists()
        and archive.exists()
        and expected_md5
        and archive.stat().st_size == total
    ):
        checksum = hashlib.md5()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() == expected_md5:
            sidecar.write_text(json.dumps(identity, indent=2) + "\n")
    bind_identity(DEST, identity)
    if archive.exists() and archive.stat().st_size != total:
        archive.rename(
            archive.with_name(archive.name + f".invalid-{uuid.uuid4().hex[:12]}")
        )
    parts = 4
    width = (total + parts - 1) // parts

    def download(i):
        low = i * width
        high = min(total - 1, (i + 1) * width - 1)
        part = DEST / f"archive.part{i}"
        size = high - low + 1
        for attempt in range(4):
            offset = part.stat().st_size if part.exists() else 0
            if offset == size:
                return part
            if offset > size:
                raise RuntimeError("Part size exceeds expected range")
            try:
                with httpx.Client(
                    follow_redirects=True, timeout=httpx.Timeout(90, connect=30)
                ) as client:
                    with client.stream(
                        "GET", URL, headers={"Range": f"bytes={low + offset}-{high}"}
                    ) as response:
                        response.raise_for_status()
                        expected = f"bytes {low + offset}-{high}/{total}"
                        if (
                            response.status_code != 206
                            or response.headers.get("Content-Range") != expected
                        ):
                            raise RuntimeError("Range integrity mismatch")
                        if etag and response.headers.get("etag") != etag:
                            raise RuntimeError("Archive changed during download")
                        with part.open("ab") as output:
                            for chunk in response.iter_bytes(1024 * 1024):
                                output.write(chunk)
                if part.stat().st_size != size:
                    raise RuntimeError("Incomplete range")
                print(f"Part {i + 1}/{parts} complete", flush=True)
                return part
            except httpx.HTTPError:
                if attempt == 3:
                    raise
        raise RuntimeError("Unfinished range")

    if not archive.exists():
        with ThreadPoolExecutor(max_workers=parts) as pool:
            paths = list(pool.map(download, range(parts)))
        tmp = archive.with_suffix(".zip.partial")
        with tmp.open("wb") as output:
            for part in paths:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
        if tmp.stat().st_size != total:
            raise RuntimeError("Archive size mismatch")
        tmp.rename(archive)
        for part in paths:
            part.unlink()
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    if expected_md5 and md5.hexdigest() != expected_md5:
        archive.rename(
            archive.with_name(archive.name + f".invalid-{uuid.uuid4().hex[:12]}")
        )
        raise ValueError("Archive does not match source cloud MD5")
    output = DEST / FILE
    try:
        raw_sha = verify_extract(archive, output, FILE, EXPECTED_RAW_BYTES)
    except (ValueError, zipfile.BadZipFile):
        archive.rename(
            archive.with_name(archive.name + f".invalid-{uuid.uuid4().hex[:12]}")
        )
        raise
    manifest = {
        "source_url": "https://www.kaggle.com/datasets/abdelwahedomar/twibot",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "third-party single tweet shard; not full TwiBot-22",
        "official_author_verified": False,
        "archive_bytes": total,
        "archive_sha256": sha.hexdigest(),
        "source_etag": etag,
        "source_cloud_md5_verified": bool(expected_md5),
        "file_sha256": raw_sha,
        "file": FILE,
        "file_bytes": output.stat().st_size,
        "zip_crc_verified": True,
        "expected_record_fields": ["id", "author_id", "created_at", "text"],
        "missing": [
            "remaining official tweet data (completeness unverified)",
            "edge.csv",
            "official release confirmation",
        ],
    }
    (DEST / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "Download/extraction verified. This is one third-party shard, not the complete benchmark.",
        flush=True,
    )


if __name__ == "__main__":
    main()
