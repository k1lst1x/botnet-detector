"""Validate/publish the data manifest and write an aggregate training-readiness report.

No training is performed. Official inputs and separate full-reference validation
must match the source identities captured before the indexing pass.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from engine.model import validate_prepared
from engine.shard_index import file_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=ROOT / "data/processed/twibot22-shard7-training-v1",
    )
    parser.add_argument(
        "--work", type=Path, default=ROOT / "data/processed/twibot22-shard7-index-v1"
    )
    parser.add_argument("--official", type=Path, default=ROOT / "data/raw/twibot22")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/reports/2026-11-19-training-data"
    )
    parser.add_argument("--arrays", type=Path)
    args = parser.parse_args()
    path = args.prepared / "manifest.json"
    manifest = json.loads(path.read_text())
    validation = validate_prepared(args.prepared)
    state = json.loads((args.work / "state.json").read_text())
    if state["phase"] != "index_complete":
        raise ValueError("Index is incomplete")
    identity = state["identity"]
    refs = json.loads((args.work / "reference-validation.json").read_text())
    audit = manifest["audit"]
    if (
        refs["source_sha256"] != audit["source_sha256"]
        or refs["records"] != audit["records"]
        or refs["invalid_reference_records"] != 0
    ):
        raise ValueError("Reference validation does not certify this source")
    user_path = args.official / "user.json"
    if (
        user_path.stat().st_size != identity["profiles_bytes"]
        or user_path.stat().st_mtime_ns != identity["profiles_mtime_ns"]
    ):
        raise ValueError("Official profile file changed since indexing")
    official = {
        name: file_hash(args.official / name)
        for name in ("label.csv", "split.csv", "user.json")
    }
    if (
        official["label.csv"] != identity["labels_sha256"]
        or official["split.csv"] != identity["splits_sha256"]
    ):
        raise ValueError("Official labels or split changed since indexing")
    recorded = manifest["source"].get("official_inputs")
    if recorded and recorded != official:
        raise ValueError("Prepared official fingerprints mismatch")
    manifest["source"]["official_inputs"] = official
    manifest["reference_validation"] = refs
    audit["deduplication_scope"] = (
        "canonical_model_input; unused raw metadata may differ"
    )
    with sqlite3.connect(f"file:{args.work}/tweets.sqlite3?mode=ro", uri=True) as db:
        counts = [r[0] for r in db.execute("SELECT n FROM author_counts ORDER BY n")]
    audit["posts_per_known_author"] = {
        "min": counts[0],
        "p10": counts[max(0, math.ceil(0.1 * len(counts)) - 1)],
        "median": counts[(len(counts) - 1) // 2],
        "p90": counts[math.ceil(0.9 * len(counts)) - 1],
        "p99": counts[math.ceil(0.99 * len(counts)) - 1],
        "max": counts[-1],
        "mean": sum(counts) / len(counts),
    }
    manifest["parquet_sha256"] = {
        f"{s}.parquet": file_hash(args.prepared / f"{s}.parquet")
        for s in ("train", "val", "test")
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix_report = None
    if args.arrays:
        matrix_report = json.loads((args.arrays / "manifest.json").read_text())
        if (
            matrix_report["source_manifest_sha256"]
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Numeric arrays belong to a different prepared manifest")
        for filename, expected in matrix_report["array_sha256"].items():
            if file_hash(args.arrays / filename) != expected:
                raise ValueError("Numeric array fingerprint mismatch")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "prepared": manifest,
                "validation": validation,
                "numeric_arrays": matrix_report,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Подготовка обучающей выборки из полного скачанного фрагмента",
        "",
        "Дата: 2026-11-19. Обучение модели **не запускалось**.",
        "Источник — один сторонний tweet-файл; это не полный официальный TwiBot-22 и не данные Threads.",
        "",
        "## Аудит всего JSON",
        "",
        f"- Прочитано записей: **{audit['records']:,}**; байт: {audit['source_bytes']:,}.",
        f"- Уникальных ID: {audit['unique_posts']:,}; повторных канонических входов: {audit['duplicate_records']:,}.",
        f"- Постов известных авторов: {audit['unique_known_posts']:,}; авторов: {audit['known_authors']:,}.",
        f"- Записей с неизвестным автором: {audit['unknown_author_records']:,}.",
        f"- Время найденных публикаций: {audit['min_timestamp']} … {audit['max_timestamp']}.",
        "- SHA256 всего tweet-файла совпал с проверенным download-manifest.",
        "- Типы referenced_tweets отдельно проверены по всему файлу; некорректных записей — 0.",
        "- Повторный ID с отличающимся каноническим входом вызывает ошибку; неиспользуемые raw-поля в dedup не сравниваются.",
        "",
        "Постов на автора (уникальные известные авторы):",
        "",
        "| min | p10 | median (lower) | p90 | p99 | max | mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            f"{v:.2f}" if isinstance(v, float) else str(v)
            for v in audit["posts_per_known_author"].values()
        )
        + " |",
        "",
        "## Отбор и исходные разбиения",
        "",
        f"Минимум {manifest['selection']['minimum_unique_posts']} постов на аккаунт; максимум {manifest['selection']['maximum_sampled_posts']} постов в расчёте признаков; seed={manifest['selection']['seed']}.",
        "Посты выбираются по минимальным BLAKE2b-хешам ID, независимо от порядка JSON, label и split.",
        "Искусственное выравнивание классов и случайное переразбиение не выполнялись.",
        f"В наборе **{manifest['rows']:,} аккаунтов**, использовано {manifest['used_posts']:,} постов.",
        "",
        "| Split | Аккаунтов | Human | Bot | Доля bot |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, c in manifest["classes_by_split"].items():
        total = c["human"] + c["bot"]
        lines.append(
            f"| {split} | {total:,} | {c['human']:,} | {c['bot']:,} | {c['bot'] / max(1, total):.2%} |"
        )
    lines += [
        "",
        "## Чувствительность к минимуму постов",
        "",
        "| Минимум | Train human | Train bot | Val human | Val bot | Test human | Test bot |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, items in audit["minimum_posts_sensitivity"].items():
        lookup = {(r["split"], r["label"]): r["accounts"] for r in items}
        lines.append(
            "| "
            + str(threshold)
            + " | "
            + " | ".join(
                str(lookup.get((s, y), 0))
                for s in ("train", "val", "test")
                for y in (0, 1)
            )
            + " |"
        )
    lines += [
        "",
        "## Покрытие признаков",
        "",
        "| Признак | Доля известных |",
        "|---|---:|",
    ]
    lines += [
        f"| {n} | {c['ratio']:.2%} |" for n, c in manifest["feature_coverage"].items()
    ]
    lines += [
        "",
        "## Ресурсы, воспроизводимость и ограничения",
        "",
        f"- Пиковый RSS исходного процесса: {manifest['peak_rss_mb']:.1f} МБ.",
        "- SQLite staging лежит на диске в data/processed, а не в tmpfs /tmp.",
        "- Сохранены SHA256 tweets, user.json, label.csv, split.csv и каждого Parquet.",
        "- Временные интенсивности/интервалы остаются null: история наблюдений неполная.",
        "- Это срез одного файла: отбор авторов и число доступных постов могут создавать смещение.",
        "- Статистики и хеши не подтверждают право называть стороннюю копию полной авторской поставкой.",
        "- Нет проверки качества на Threads, реального training run и оценённого модельного артефакта.",
        "",
        "## Следующий шаг",
        "",
        "Матрицы для структурного baseline подготовлены отдельно. Следующий этап — подключить их к training entry point и обучить контрольный baseline и XGBoost.",
        "Fit и импутация — только train; параметры и порог — val; test использовать после фиксации решения.",
        "Для первого baseline предложены девять структурных признаков/URL и маски; типы публикаций пока исключены из-за низкого покрытия. Не обучаться на IDs/split/label как признаках.",
        "Отчёт должен включать precision/recall/F1/PR-AUC, confusion matrix, покрытие и ограничения переноса.",
        "Проверку на настоящих Threads проводить отдельно.",
        "",
        "```bash",
        "python -m scripts.prepare_training_shard",
        "# Для нового отбора из готового индекса — --resume-index и НОВЫЙ --output:",
        "python -m scripts.prepare_training_shard --resume-index --output data/processed/new-selection",
        "python -m engine.cli validate data/processed/twibot22-shard7-training-v1",
        "python -m scripts.report_training_shard",
        "```",
        "",
    ]
    if matrix_report:
        position = lines.index("## Ресурсы, воспроизводимость и ограничения")
        matrix_lines = [
            "## Компактный вход для первого обучения",
            "",
            f"Каталог: {args.arrays.name}. Девять признаков + девять масок = 18 столбцов float32.",
            "Метки хранятся отдельно в uint8; ID и split не попадают в X.",
            "NaN сохранены; импутация, масштабирование и модель ещё не обучались.",
            f"Суммарный размер шести NPY-файлов: {sum((args.arrays / n).stat().st_size for n in matrix_report['array_sha256']):,} байт.",
            "NPY читаются с mmap_mode='r', allow_pickle=False. Порядок строк совпадает с исходным Parquet.",
            "",
        ]
        lines[position:position] = matrix_lines
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "rows": manifest["rows"],
                "counts": manifest["counts"],
                "classes": manifest["classes_by_split"],
                "report": str(args.output.with_suffix(".md")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
