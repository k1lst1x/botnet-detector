# Подготовка и запуск движка

Подготовлены два приложения: Django (пользователи и CRM) и FastAPI + Celery
(импорт, сбор, признаки, распознавание). Обучение запускается отдельной CLI-командой.
Ни API, ни worker, ни веб-интерфейс автоматически модель не обучают.

## Что уже подготовлено

- data/raw/twibot22: четыре файла Zenodo, 824 609 718 байт, опубликованные MD5 проверены.
- data/processed/twibot22-profiles: миллион строк Parquet, 700000/200000/100000 train/val/test.
- Полная проверка строк и непересечения ID выполнена; модель не обучалась.
- Метаданные загрузки: plans/twibot22-manifest.json.
- 31 числовой признак + 31 маска пропусков: [формулы](features.md).
- Историческая дата снимка и посты отсутствуют, поэтому возраст и поведение не выдумываются.

Эти данные хранятся локально и исключены из Git. На другом компьютере их нужно
скачать заново из https://zenodo.org/records/7012904. Полные tweet_*.json и edge.csv
нужно запросить у авторов через https://github.com/LuoUndergradXJTU/TwiBot-22.
Без полного набора можно обучить только профильный baseline, а качество его
переноса на Threads ещё не подтверждено.

## Локальный запуск

Python 3.12+; текущие проверки выполнены на Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

В .env задайте ENGINE_API_KEY и DJANGO_SECRET_KEY случайными строками,
POSTGRES_PASSWORD — отдельным паролем. Генератор: `python -c 'import secrets; print(secrets.token_hex(32))'`.
.env не передаётся в Git. Для запуска вне Compose переменные нужно экспортировать
в окружение терминала; Python-приложения сами .env не загружают.

При установленном Redis запустите три процесса с одинаковыми ENGINE_* и CELERY_BROKER_URL:

```bash
uvicorn service.app:create_app --factory --port 8001
celery -A service.worker:celery worker --loglevel=INFO --concurrency=2
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 8000
```

Django: http://127.0.0.1:8000/engine/; вход: /auth/login/; админка: /admin/.
FastAPI OpenAPI: http://127.0.0.1:8001/docs.
Первый `migrate` выполняется один раз, затем запуск сервера.

## Docker Compose (локальная разработка)

```bash
cp .env.example .env
# Заполнить три секрета в .env.
docker compose up --build -d
docker compose exec web python manage.py createsuperuser
```

Две PostgreSQL-базы: botnet_crm для Django и botnet_engine для FastAPI.
Redis хранит очередь; jobs сохраняются в SQL, сырые ответы и нормализованные
снимки — в JSON задания. Это начальное хранилище для небольших импортов; большие
исследовательские данные идут отдельным путём Parquet и не импортируются в CRM.
Загружать миллион профилей через веб не нужно.

Порты опубликованы только на localhost. Compose использует Django runserver и
предназначен для локальной разработки. Для внешнего сервера потребуются production
WSGI/ASGI, HTTPS и обслуживание static. Большие raw/processed файлы не входят в образ.
Образ и PostgreSQL/Redis запускаются пользователем; в ходе подготовки проверена
конфигурация Compose, но полноценный запуск контейнеров не выполнялся.

## Проверка цепочки без модели

1. Войти в Django /engine/.
2. Импортировать data/threads_example.json как JSON (или CSV/JSONL по формату ниже).
3. Открыть задание и нажать «Обновить состояние» до completed.
4. Нажать «Проверить аккаунты», затем обновить состояние нового задания.
5. Увидеть подготовленные признаки и «Модель ещё не обучена».
6. В /engine/accounts/ добавить ручную метку и заметки. Они отделены от решения ML.

Задания и CRM разделены по пользователям. Django сам задаёт X-Tenant-ID=user:<pk>;
браузер не получает ключ FastAPI и не выбирает чужой tenant. FastAPI — внутренний
сервис: один ENGINE_API_KEY доверяет вызывающему серверу устанавливать tenant.
Не выдавайте этот ключ конечным пользователям и не используйте tenant-заголовок
как аутентификацию при прямом публичном доступе.

Старые страницы /accounts/, /graph/, /data/ показывают прежний общий демонстрационный
набор. Рабочий пользовательский поток — /engine/ и /engine/accounts/. Старый pipeline
оставлен для исследований; без артефакта он теперь завершает работу с model_not_ready,
а веб-кнопка /analyze/ перенаправляет в новый движок.

## Входные форматы

CSV/JSONL: одна строка — одна публикация. Обязательны author_id, post_id,
created_at с часовым поясом. platform по умолчанию threads; text может быть null.
Дополнительно username/handle, followers, following, biography,
is_reply, is_repost, is_quote, has_media. Пустой boolean = неизвестен,
true/false/1/0 — известен. Противоречивые профили и повторяющиеся post_id отклоняются.
Если передан только handle/username, используется отдельный ID username:…
с явной пометкой нестабильной идентичности, как в keyword search.
Для публичных ID используйте стабильные идентификаторы платформы, а не username.

```csv
platform,author_id,post_id,created_at,text,is_reply
threads,123,p1,2026-11-01T08:00:00Z,Привет,false
threads,123,p2,2026-11-01T09:00:00Z,Новость,
```

CSV/JSONL не объявляют полную историю. JSON принимает Dataset из
engine/contracts.py: accounts, posts, observation. Полноту complete=true можно
указывать только для известного полного окна. Время приводится к UTC; посты вне
окна отклоняются. Полный пример — data/threads_example.json.

Лимит веб/API: 8 МБ текста, 10000 аккаунтов / 100000 постов; большие файлы разбивать.
Встроенная схема доступна через FastAPI OpenAPI. Для крупного внешнего развёртывания
лимит HTTP-тела нужно дополнительно задать в reverse proxy.

## Подготовка TwiBot-22 (без обучения)

```bash
python -m engine.cli prepare --source data/raw/twibot22 --output data/processed/twibot22-profiles
python -m engine.cli validate data/processed/twibot22-profiles
```

Выходной каталог должен быть пуст. Для повторного запуска выберите новый каталог.
Для smoke-теста можно добавить --limit 10000; это не заменяет полный датасет.
Reader обрабатывает JSON-массивы через ijson, CSV через потоковый reader, joins —
через дисковую SQLite, Parquet пишет батчами. Имена split: train/val/test;
также принимается valid как синоним val. Неизвестные split/label, дубли и несовпадения
ID приводят к ошибке. Manifest записывается только после успешного завершения;
частичные Parquet после ошибки не являются готовым набором.

После получения полных данных положите tweet_*.json и edge.csv рядом с user.json
и подготовьте новый каталог. Связь tweet→author берётся из author_id или post/own
рёбер. На очень активный аккаунт его посты пока загружаются вместе; общий набор
остаётся на диске. Исторический архив сам по себе не доказывает полное окно
наблюдения: временные признаки по умолчанию остаются null. Для основного
переносимого временного эксперимента потребуется подтверждённое окно и повторная
подготовка с явно описанной полнотой, а не догадка по min/max публикаций.
Для этого prepare принимает --observation window.json со схемой Observation
(start/end/source/complete/truncated); публикации вне объявленного окна отклоняются.

## Обучение позже, по отдельному решению

Подготовлен `python -m engine.cli train --help`, но обучение в этой работе не запускалось.
Команда требует --prepared и --output; --feature-set profile|portable, --seed,
--min-precision. Portable не принимает профильный набор без tweets.
Обучение использует только train, выбирает порог по val, не читает test и не
переобучает модель на объединённых данных. Медианная импутация входит в артефакт.
Если заданный precision на validation недостижим, сохраняется явный флаг и
консервативный порог; это не гарантия precision на Threads.

Для финальной отдельной оценки подготовлена команда evaluate --prepared … --model …:
precision/recall/F1 bot, macro-F1, PR-AUC, ROC-AUC, Brier и confusion matrix.
Артефакт содержит признаки, версию схемы, seed, порог, метрики validation и версии
зависимостей. Это baseline с некалиброванным score; до контрольной разметки Threads
его нельзя объявлять точной вероятностью бота. Подбор гиперпараметров, калибровка,
ablation и оценка переноса — следующий исследовательский этап.

FastAPI/worker читают ENGINE_MODEL_PATH (по умолчанию media/models/portable.joblib).
Артефакт должен быть доверенным: joblib может выполнять Python при загрузке.
Загрузки пользовательских моделей через API нет. После установки новой модели
перезапустите API и workers. Подготовлены проверки отсутствующего/несовместимого
артефакта, но полноценный прогноз настоящей модели проверяется после её обучения.

## Threads

Клиент engine/threads.py использует официальный keyword_search и Bearer token,
а не браузерный парсинг. THREADS_ACCESS_TOKEN пока не настроен; job возвращает
threads_not_configured. Адаптер проверен fixture-транспортом, не реальным API.
Перед подключением проверить доступные поля и разрешения по актуальной Meta-документации:
https://developers.facebook.com/documentation/threads/keyword-search

Пагинация следует только курсору на фиксированном API host, не произвольному next URL.
Повторы ограничены тремя попытками для транспортных ошибок, 429 и 5xx; задержка ≤30с.
Курсоры и очищенные страницы сохраняются в jobs.checkpoint, token и paging.next — нет.
В статусе задания доступен next_cursor. Параметр after позволяет продолжить сбор
новым заданием; страницы исходного задания остаются в его хранилище.

## Состояния и восстановление

queued → running → completed или failed. Повторная доставка завершённого задания
не пересчитывает его. Idempotency-Key привязан к tenant и типу операции;
другой payload с тем же ключом даёт 409. Публикация в брокер идёт после SQL commit:
при сбое сохраняется queued и возвращается 503 с job_id. Django сохраняет
ссылку на такое задание и показывает предупреждение об очереди вместо нового импорта.

```bash
python -m service.worker --recover
```

Команда повторно публикует queued и возвращает running с истёкшей часовой арендой
в очередь. Каждая попытка получает собственный lease token: старый worker
не может сохранить результат или checkpoint новой попытки. Её можно запускать
периодически оператором после сбоя. Это не скрытый
бесконечный retry: failed требует нового задания. Сбор при повторном выполнении
заново читает начальный курсор задания; для явного продолжения используйте after.

liveness 200 означает живой API, readiness 503/model_not_ready — нет обученной модели;
импорт и подготовка при этом работают. insufficient_data — слишком мало реально
наблюдаемых признаков требуемого моделью набора; missing masks не считаются данными.
Координация возвращается отдельно и не смешивается со score.

## Проверки

```bash
pip install -r requirements-dev.txt
pytest -q
python manage.py check
python manage.py makemigrations --check --dry-run
```

Два старых теста с fit на синтетике выключены без RUN_LEGACY_TRAINING_TESTS=1.
Обычный pytest не обучает модель. TestClient может требовать разрешённые локальные
сокеты в sandbox. Проверки покрывают контракты, признаки, tiny TwiBot→Parquet,
изолированный API→SQL job→worker, Celery eager с memory broker, адаптер Threads
и Django CRM. Отдельного живого Redis/PostgreSQL интеграционного запуска пока нет.

Финальный прогон подготовки: **38 passed, 2 skipped** (только старые обучающие тесты).
Django check, migration drift, Ruff и Docker Compose config прошли.
