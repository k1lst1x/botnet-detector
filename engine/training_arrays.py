"""Batch export of selected numeric columns; leaves all preprocessing unfitted."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .features import BASE_FEATURES, FEATURE_VERSION
from .shard_index import file_hash


def export_training_arrays(prepared, output, base_features):
    prepared, output = Path(prepared), Path(output)
    if (
        not base_features
        or len(set(base_features)) != len(base_features)
        or not set(base_features) <= set(BASE_FEATURES)
    ):
        raise ValueError("Unsafe or unknown feature subset")
    names = list(base_features) + [f"{name}__missing" for name in base_features]
    source_manifest = json.loads((prepared / "manifest.json").read_text())
    if source_manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("Feature version mismatch")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Array output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    splits = {}
    total = 0
    hashes = {}
    for split in ("train", "val", "test"):
        parquet = pq.ParquetFile(prepared / f"{split}.parquet")
        n = parquet.metadata.num_rows
        if n < 1 or n != source_manifest["counts"][split]:
            raise ValueError("Empty or mismatched split")
        X = np.lib.format.open_memmap(
            output / f"{split}.X.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, len(names)),
        )
        y = np.lib.format.open_memmap(
            output / f"{split}.y.npy", mode="w+", dtype=np.uint8, shape=(n,)
        )
        offset = 0
        classes = Counter()
        for batch in parquet.iter_batches(
            batch_size=8192, columns=["external_user_id", "split", "label"] + names
        ):
            values = {
                name: batch.column(batch.schema.get_field_index(name))
                for name in batch.schema.names
            }
            ids = values["external_user_id"].to_pylist()
            if any(
                not isinstance(uid, str) or not uid or uid in seen for uid in ids
            ) or len(set(ids)) != len(ids):
                raise ValueError("Duplicate/cross-split identity")
            seen.update(ids)
            if any(s != split for s in values["split"].to_pylist()):
                raise ValueError("Wrong split membership")
            labels = values["label"].to_numpy(zero_copy_only=False)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("Invalid class labels")
            block = np.column_stack(
                [values[name].to_numpy(zero_copy_only=False) for name in names]
            ).astype(np.float64)
            if np.isinf(block).any():
                raise ValueError("Infinite feature")
            if np.any(np.abs(block[np.isfinite(block)]) > np.finfo(np.float32).max):
                raise ValueError("Feature exceeds float32 range")
            masks = block[:, len(base_features) :]
            if not np.array_equal(
                masks, np.isnan(block[:, : len(base_features)]).astype(float)
            ):
                raise ValueError("Missing-mask mismatch")
            count = batch.num_rows
            X[offset : offset + count] = block.astype(np.float32)
            y[offset : offset + count] = labels.astype(np.uint8)
            classes.update(int(v) for v in labels)
            offset += count
        if offset != n:
            raise ValueError("Row count mismatch")
        X.flush()
        y.flush()
        del X, y
        splits[split] = {"rows": n, "classes": {"human": classes[0], "bot": classes[1]}}
        for kind in ("X", "y"):
            filename = f"{split}.{kind}.npy"
            hashes[filename] = file_hash(output / filename)
        total += n
    if total != source_manifest["rows"]:
        raise ValueError("Manifest total mismatch")
    report = {
        "feature_version": FEATURE_VERSION,
        "feature_names": names,
        "dtype": "float32",
        "label_dtype": "uint8",
        "splits": splits,
        "rows": total,
        "preprocessing_fitted": False,
        "training_performed": False,
        "missing_values": "NaN, not imputed; fit preprocessing on train only",
        "identity_mapping": "Same row order as external_user_id in the corresponding source Parquet; ID is not a model feature",
        "source_manifest_sha256": hashlib.sha256(
            (prepared / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_parquet_sha256": {
            f"{s}.parquet": file_hash(prepared / f"{s}.parquet")
            for s in ("train", "val", "test")
        },
        "array_sha256": hashes,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
