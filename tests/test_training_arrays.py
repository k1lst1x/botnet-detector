import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from engine.training_arrays import export_training_arrays


def prepared(tmp_path, duplicate=False):
    root = tmp_path / "prepared"
    root.mkdir()
    for i, split in enumerate(("train", "val", "test")):
        row = {
            "external_user_id": "same" if duplicate else f"u{i}",
            "split": split,
            "label": i % 2,
            "duplicate_ratio": None if split == "train" else 0.5,
            "duplicate_ratio__missing": 1.0 if split == "train" else 0.0,
        }
        schema = pa.schema(
            [
                ("external_user_id", pa.string()),
                ("split", pa.string()),
                ("label", pa.int8()),
                ("duplicate_ratio", pa.float64()),
                ("duplicate_ratio__missing", pa.float64()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist([row], schema=schema), root / f"{split}.parquet"
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "feature_version": "portable-v1",
                "rows": 3,
                "counts": {"train": 1, "val": 1, "test": 1},
            }
        )
    )
    return root


def test_numeric_arrays_preserve_nan_masks_and_split_labels_without_fitting(tmp_path):
    root = prepared(tmp_path)
    report = export_training_arrays(root, tmp_path / "arrays", ["duplicate_ratio"])
    X = np.load(tmp_path / "arrays/train.X.npy", mmap_mode="r", allow_pickle=False)
    assert X.shape == (1, 2) and X.dtype == np.float32
    assert np.isnan(X[0, 0]) and X[0, 1] == 1
    assert np.load(tmp_path / "arrays/val.y.npy", allow_pickle=False).tolist() == [1]
    assert np.load(tmp_path / "arrays/test.X.npy", allow_pickle=False).tolist() == [
        [0.5, 0.0]
    ]
    assert report["feature_names"] == ["duplicate_ratio", "duplicate_ratio__missing"]
    assert report["preprocessing_fitted"] is False


def test_leaking_identities_and_labels_as_features_are_rejected(tmp_path):
    root = prepared(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="identity"):
        export_training_arrays(root, tmp_path / "arrays", ["duplicate_ratio"])
    assert not (tmp_path / "arrays/manifest.json").exists()
    with pytest.raises(ValueError, match="feature"):
        export_training_arrays(root, tmp_path / "unsafe", ["label"])
