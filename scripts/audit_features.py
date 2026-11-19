"""Report feature coverage for prepared Twitter and an explicitly identified input.

python -m scripts.audit_features --help
This is a data audit, never training or a claim of target-platform quality.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq

from engine.datasets import read_import
from engine.features import (
    BASE_FEATURES,
    FEATURE_VERSION,
    PROFILE_FEATURES,
    extract_features,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path, default=ROOT / "data/processed/twibot22-profiles"
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/threads_example.json"
    )
    parser.add_argument("--format", choices=["json", "jsonl", "csv"], default="json")
    parser.add_argument(
        "--input-kind", choices=["synthetic_example", "real_export"], required=True
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/reports/2026-11-19-feature-audit"
    )
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("Prepared schema mismatch")
    totals = {name: 0 for name in BASE_FEATURES}
    split_counts = {}
    for split in ("train", "val", "test"):
        count = 0
        for batch in pq.ParquetFile(args.prepared / f"{split}.parquet").iter_batches(
            columns=BASE_FEATURES
        ):
            count += batch.num_rows
            for name in BASE_FEATURES:
                column = batch.column(batch.schema.get_field_index(name))
                totals[name] += batch.num_rows - column.null_count
        if count != manifest["counts"][split]:
            raise ValueError("Manifest count mismatch")
        split_counts[split] = count
    n = sum(split_counts.values())
    data = read_import(args.input.read_text(), args.format)
    posts = defaultdict(list)
    for post in data.posts:
        posts[(post.platform, post.author_id)].append(post)
    target = {name: 0 for name in BASE_FEATURES}
    for account in data.accounts:
        features = extract_features(account, posts[account.key], data.observation)
        for name in BASE_FEATURES:
            target[name] += features[name] is not None
    m = len(data.accounts)
    if not m:
        raise ValueError("Target input has no accounts")
    rows = [
        {
            "name": name,
            "group": "profile" if name in PROFILE_FEATURES else "portable",
            "twitter_observed": totals[name],
            "twitter_ratio": totals[name] / max(1, n),
            "input_observed": target[name],
            "input_ratio": target[name] / m,
        }
        for name in BASE_FEATURES
    ]
    common = [
        row["name"]
        for row in rows
        if row["twitter_ratio"] >= 0.5 and row["input_ratio"] >= 0.5
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "prepared_rows": n,
        "split_rows": split_counts,
        "behavior_available": manifest["behavior_available"],
        "input_kind": args.input_kind,
        "input_accounts": m,
        "input_posts": len(data.posts),
        "real_threads_quality_validated": False,
        "common_features_at_50_percent_coverage": common,
        "features": rows,
        "limitations": [
            "Target quality requires independently labelled Threads data.",
            "Synthetic example demonstrates contract coverage only, not real API availability.",
            "Coverage is not feature importance and is not a licence to change missingness rules.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Аудит покрытия признаков Twitter / входного файла",
        "",
        f"Дата UTC: {report['created_at_utc']}. Версия: {FEATURE_VERSION}.",
        f"Профили Twitter: {n:,}; train/val/test: {split_counts}.",
        f"Вход: **{args.input_kind}**, аккаунтов {m}, публикаций {len(data.posts)}.",
        "**Это не проверка качества модели и не доказательство доступности реального Threads API.**",
        "",
        "| Признак | Группа | Twitter: известен | Вход: известен |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {r['name']} | {r['group']} | {r['twitter_ratio']:.1%} | {r['input_ratio']:.1%} |"
        for r in rows
    )
    lines += [
        "",
        "Пересечение с покрытием ≥50% на каждой стороне: "
        + (", ".join(common) if common else "**отсутствует**")
        + ".",
        "",
        (
            "Есть пересечение наблюдаемых признаков; проверить состав, качество и переносимость."
            if common
            else "Вывод: у выбранных входов нет пересечения признаков с указанным покрытием."
        ),
        "Совместимость форматов сама по себе не доказывает пригодность модели.",
        "Проверить источники данных, недостающие поля и сопоставимость процедуры отбора.",
        "Качество и пригодность набора определяются отдельным экспериментом.",
        "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("features", "limitations")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
