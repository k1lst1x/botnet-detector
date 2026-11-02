import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase
from django.core.management import call_command

from analysis.twibot import (
    FEATURES,
    TwiBotFormatError,
    load_dataset,
    profile_features,
    save_artifacts,
    train_baseline,
    validate_dataset,
)


def _user(index, bot):
    return {
        "id": f"u{index}",
        "username": f"account_{index}",
        "name": f"Account {index}",
        "description": "" if bot else "personal profile",
        "created_at": "2020-01-01T00:00:00Z" if bot else "2012-01-01T00:00:00Z",
        "location": None if bot else "Almaty",
        "verified": False,
        "protected": False,
        "profile_image_url": None if bot else "https://example.test/avatar.jpg",
        "public_metrics": {
            "followers_count": 5 if bot else 500,
            "following_count": 900 if bot else 120,
            "tweet_count": 10000 if bot else 800,
            "listed_count": 0 if bot else 8,
        },
    }


def _write_fixture(root: Path):
    users = [_user(i, bot=i % 2 == 1) for i in range(12)]
    (root / "user.json").write_text(json.dumps(users), encoding="utf-8")
    with (root / "label.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label"])
        writer.writerows((user["id"], "bot" if i % 2 else "human") for i, user in enumerate(users))
    with (root / "split.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "split"])
        for i, user in enumerate(users):
            writer.writerow([user["id"], "train" if i < 6 else "valid" if i < 9 else "test"])


class TwiBotTests(SimpleTestCase):
    def test_profile_features_are_language_independent_and_complete(self):
        features = profile_features(_user(1, True))
        self.assertEqual(set(features), set(FEATURES))
        self.assertEqual(features["tweet_count"], 10000)
        self.assertLess(features["followers_following_ratio"], 0.01)

    def test_official_splits_are_preserved_during_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            rows = load_dataset(root)
            self.assertEqual([len(rows[name]) for name in ("train", "valid", "test")], [6, 3, 3])
            self.assertFalse({item["id"] for item in rows["train"]} & {item["id"] for item in rows["test"]})
            bundle, metrics = train_baseline(rows, kind="sklearn", seed=1)
            self.assertEqual(metrics["split_source"], "official split.csv")
            self.assertEqual(metrics["test"]["n"], 3)
            output = root / "artifacts"
            save_artifacts(bundle, metrics, output)
            self.assertTrue((output / "model.joblib").is_file())
            self.assertTrue((output / "metrics.json").is_file())

    def test_missing_required_file_is_reported(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TwiBotFormatError, "missing TwiBot-22 files"):
                validate_dataset(Path(directory))

    def test_management_command_writes_reproducible_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            output = root / "command-output"
            stdout = io.StringIO()
            call_command(
                "train_twibot22_baseline",
                dataset=str(root),
                output=str(output),
                model="sklearn",
                seed=7,
                stdout=stdout,
            )
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["model"], "sklearn-hist")
            self.assertIn("Артефакты", stdout.getvalue())
