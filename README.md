# botnet/detector

Исследовательская система для обнаружения автоматизированных и скоординированных
аккаунтов в Threads. Дипломный проект.

> Оценка модели — сигнал для ручной проверки, а не доказательство автоматизации.

## Архитектура

```
Django CRM ──► FastAPI (внутренний) ──► Celery + Redis ──► импорт / сбор
                                                         ──► признаки ──► модель ──► CRM
```

- **Django** — вход, импорт файлов, задания, CRM с ручной разметкой (`/engine/`, `/engine/accounts/`).
- **FastAPI** — внутренний API импорта, сбора и анализа (`service/`).
- **engine/** — контракты данных, признаки, обучение, инференс, адаптер Threads.
- **scripts/** — воспроизводимые эксперименты; отчёты в `docs/reports/`.

## Текущее состояние (21.11.2026)

- Активная модель: profile-XGBoost в режиме очереди проверки
  ([отчёт](docs/reports/2026-11-20-review-mvp.md)). Validation: precision 60,4%, recall 19,9%
  при очереди 8,4% аккаунтов. Независимой проверки на Threads пока нет.
- Обучающие данные: фрагмент TwiBot-22, 197 413 аккаунтов
  ([отчёт](docs/reports/2026-11-19-training-data.md)).
- Попытки улучшения (посты, метаданные, веса классов, межаккаунтный граф) не превзошли
  профильную модель: [сравнение](docs/reports/2026-11-20-feature-model-comparison.md),
  [ошибки](docs/reports/2026-11-20-error-driven.md),
  [метаданные](docs/reports/2026-11-20-recovered-metadata.md),
  [graph-v6](docs/reports/2026-11-20-graph-v6.md).
- Threads API без доступа: работает импорт CSV/JSONL/JSON.

## Модели

Файлы моделей не входят в Git. Скачивать их нужно из папки
[Google Drive](https://drive.google.com/drive/folders/1JLmfTDU9aPYOnZ49P_aW9yg1Tjy9DILI), обновления загружать туда же.
Установка, контрольные суммы и порядок обновления: [docs/MODELS.md](docs/MODELS.md).

## Запуск

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
cp -n .env.example .env   # заполнить секреты
# скачать модель с Google Drive и подключить: docs/MODELS.md
docker compose up --build -d
docker compose exec web python manage.py createsuperuser
```

Подробно: [docs/engine.md](docs/engine.md) · [признаки](docs/features.md) ·
[данные](docs/DATA_ACCESS.md) · [план MVP](MVP_HANDOFF.md).

Прежнее демо на синтетике (Louvain, граф W(A,B)) удалено; код доступен по git-тегу
`legacy-demo-final`.
