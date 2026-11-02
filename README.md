# botnet/detector

Исследовательская система для **обнаружения скоординированных бот-сетей в Threads** (казахстанский и русскоязычный сегмент).
Дипломный проект: ML по отдельным аккаунтам + графовый анализ связей и синхронности.

> Синхронность — индикатор для проверки, а не доказательство автоматизации.

## Метод

```
сбор/импорт → признаки (поведение, контент, время) → ML: P(bot) + Isolation Forest
            → взвешенный граф W(A,B) → Louvain / DBSCAN → оценка групп → объяснение
```

**W(A,B) = 0.30·TextSim + 0.25·TimeSim + 0.20·URLSim + 0.10·HashtagSim + 0.15·InteractionSim**

| компонент | расчёт |
|---|---|
| TextSim | cosine TF-IDF-центроидов аккаунтов (1–2-граммы) |
| TimeSim | доля публикаций, совпавших в окне Δt (±1 бакет) |
| URLSim / HashtagSim / InteractionSim | Jaccard множеств с затуханием для малых множеств |

Риск аккаунта: `0.5·P(bot) + 0.2·anomaly + 0.3·coordination_score(сообщества)`.

Модели: XGBoost → CatBoost → sklearn GradientBoosting (автовыбор, что установлено), Isolation Forest,
Louvain (NetworkX), DBSCAN по расстоянию `1 − W`.

### Признаки

- **поведение/контент (16):** posts_per_day, avg_post_interval, interval_cv, night_activity_ratio, hour_entropy,
  max_active_streak_h, repost_ratio, reply_ratio, url_ratio, hashtag_ratio, duplicate_ratio,
  self_text_similarity, text_diversity, interaction_rate, burstiness, max_posts_per_hour
- **сетевые (14):** degree, in/out_degree, clustering, avg_common_neighbors, reciprocity, pagerank,
  repeat_interaction_ratio, coord_degree, coord_strength, sync_score, community_size, community_density, edge_stability

Отсутствующие значения → NaN → импутация медианой.

### Эксперименты

| | признаки | метрики | где |
|---|---|---|---|
| A | профиль + поведение | Precision, Recall, F1, PR-AUC | `metrics.A_baseline` |
| B | A + сетевые | то же | `metrics.B_extended` |
| C | B + граф + сообщества | Group P/R, ARI, NMI | `metrics.C_groups` |
| D | обучение на X → Threads | падение качества | импорт X → train, импорт Threads → `run_analysis` |

## Стек

Python **3.12** · Django **5.2 LTS** · scikit-learn · XGBoost · NetworkX · SQLite (dev) / PostgreSQL · vis-network

## Запуск

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py generate_synthetic --reset     # ~300 аккаунтов, 5 кампаний
python manage.py run_analysis --retrain
python manage.py runserver                      # http://127.0.0.1:8000
```

PostgreSQL: задайте `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST` и установите `psycopg[binary]`.

### Команды

| команда | что делает |
|---|---|
| `generate_synthetic [--reset --seed N]` | синтетические сценарии: люди, одиночные боты, кампании, «трудные негативы» |
| `import_csv --accounts a.csv --posts p.csv` | импорт Kaggle/Cresci/TwiBot/Threads ([формат](data/README.md)) |
| `fetch_threads <запросы>` | сбор через официальный Threads API (keyword search) |
| `run_analysis [--retrain --model xgboost --threshold 0.3 --window 600]` | полный pipeline |
| `train_twibot22_baseline --dataset <dir>` | profile baseline на официальных split TwiBot-22 ([инструкция](data/TWIBOT.md)) |

### Веб-интерфейс и API

`/` обзор и метрики · `/accounts/` · `/accounts/<handle>/` объяснение оценки · `/communities/` · `/graph/` интерактивный граф · `/data/` импорт

JSON: `/api/accounts/`, `/api/accounts/<handle>/`, `/api/communities/`, `/api/graph/?min_weight=0.4`

## Структура

```
analysis/      чистый Python: признаки, граф, ML, синтетика, pipeline
core/          Django-приложение: модели, импорт, views, шаблоны, команды
docs/          лендинг (GitHub Pages)
data/          формат и примеры CSV
tests/         python manage.py test tests
```

## GitHub Pages

Workflow `.github/workflows/pages.yml` публикует папку `docs/`. В репозитории: **Settings → Pages → Source: GitHub Actions**.
После первого успешного deploy лендинг будет доступен по адресу
[`https://k1lst1x.github.io/botnet-detector/`](https://k1lst1x.github.io/botnet-detector/).

## Ограничения (честно)

- На синтетике модель почти идеальна (F1≈1) — это проверка корректности pipeline, а не результат исследования.
- Threads API даёт не все связи (нет подписок), поэтому граф строится по контенту, времени и упоминаниям.
- Веса W и порог ребра нужно подбирать на валидации, а не брать по умолчанию.

## План (12 недель)

- [x] каркас MVP: импорт, признаки, ML baseline, граф, сообщества, дашборд
- [x] TwiBot-22: потоковый загрузчик, проверка формата и языконезависимый profile baseline
- [ ] реальные датасеты: полный прогон TwiBot-22, Cresci-2017, TwiBot-20 и Kaggle Twitter
- [ ] эмбеддинги (multilingual-e5 / LaBSE) вместо TF-IDF для TextSim
- [ ] подбор весов W на валидации, Leiden
- [ ] сбор Threads KZ/RU, разметка групп, эксперимент D
- [ ] фоновые задачи (Celery) для больших выборок
