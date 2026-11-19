# Аудит покрытия признаков Twitter / входного файла

Дата UTC: 2026-11-19T12:25:05.068935+00:00. Версия: portable-v1.
Профили Twitter: 1,000,000; train/val/test: {'train': 700000, 'val': 200000, 'test': 100000}.
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
| duplicate_ratio | portable | 0.0% | 100.0% |
| text_length_median | portable | 0.0% | 100.0% |
| text_length_cv | portable | 0.0% | 100.0% |
| text_digit_ratio | portable | 0.0% | 100.0% |
| text_space_ratio | portable | 0.0% | 100.0% |
| text_punctuation_ratio | portable | 0.0% | 100.0% |
| url_ratio | portable | 0.0% | 100.0% |
| unique_domains | portable | 0.0% | 100.0% |
| domain_concentration | portable | 0.0% | 100.0% |
| unique_targets | portable | 0.0% | 0.0% |
| target_concentration | portable | 0.0% | 0.0% |

Пересечение с покрытием ≥50% на каждой стороне: **отсутствует**.

Вывод: для текущего профильного Twitter-набора и примера публикаций нельзя считать
модель совместимой только потому, что форматы успешно читаются. Нужны публикации
Twitter либо реальное обогащение профиля Threads, затем повторный аудит.
Качество и пригодность набора определяются отдельным экспериментом.
