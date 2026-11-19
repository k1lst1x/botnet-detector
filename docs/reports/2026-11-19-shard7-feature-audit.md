# Аудит покрытия признаков Twitter / входного файла

Дата UTC: 2026-11-19T12:38:35.734180+00:00. Версия: portable-v1.
Профили Twitter: 406; train/val/test: {'train': 349, 'val': 34, 'test': 23}.
Вход: **synthetic_example**, аккаунтов 1, публикаций 2.
**Это не проверка качества модели и не доказательство доступности реального Threads API.**

| Признак | Группа | Twitter: известен | Вход: известен |
|---|---|---:|---:|
| followers_log1p | profile | 100.0% | 0.0% |
| following_log1p | profile | 100.0% | 0.0% |
| followers_following_ratio | profile | 100.0% | 0.0% |
| bio_present | profile | 100.0% | 0.0% |
| avatar_present | profile | 100.0% | 0.0% |
| account_age_days | profile | 0.0% | 0.0% |
| posts_per_day | portable | 0.0% | 0.0% |
| interval_median_s | portable | 0.0% | 0.0% |
| interval_p90_s | portable | 0.0% | 0.0% |
| interval_cv | portable | 0.0% | 0.0% |
| burstiness | portable | 0.0% | 0.0% |
| short_interval_ratio | portable | 0.0% | 0.0% |
| hour_entropy | portable | 0.0% | 0.0% |
| circadian_amplitude | portable | 0.0% | 0.0% |
| max_posts_per_hour | portable | 0.0% | 0.0% |
| active_day_ratio | portable | 0.0% | 0.0% |
| reply_ratio | portable | 0.0% | 100.0% |
| repost_ratio | portable | 0.0% | 0.0% |
| quote_ratio | portable | 0.0% | 0.0% |
| media_ratio | portable | 0.0% | 0.0% |
| duplicate_ratio | portable | 100.0% | 100.0% |
| text_length_median | portable | 100.0% | 100.0% |
| text_length_cv | portable | 94.3% | 100.0% |
| text_digit_ratio | portable | 100.0% | 100.0% |
| text_space_ratio | portable | 100.0% | 100.0% |
| text_punctuation_ratio | portable | 100.0% | 100.0% |
| url_ratio | portable | 100.0% | 100.0% |
| unique_domains | portable | 100.0% | 100.0% |
| domain_concentration | portable | 90.9% | 100.0% |
| unique_targets | portable | 0.0% | 0.0% |
| target_concentration | portable | 0.0% | 0.0% |

Пересечение с покрытием ≥50% на каждой стороне: duplicate_ratio, text_length_median, text_length_cv, text_digit_ratio, text_space_ratio, text_punctuation_ratio, url_ratio, unique_domains, domain_concentration.

Есть пересечение наблюдаемых признаков; проверить состав, качество и переносимость.
Совместимость форматов сама по себе не доказывает пригодность модели.
Проверить источники данных, недостающие поля и сопоставимость процедуры отбора.
Качество и пригодность набора определяются отдельным экспериментом.
