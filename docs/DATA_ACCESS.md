# Данные TwiBot-22: что действительно скачано

Проверено 2026-11-19. Обучение не запускалось.

## Официальная поставка

В [репозитории авторов](https://github.com/LuoUndergradXJTU/TwiBot-22#how-to-download-twibot-22-dataset)
для полного набора указан запрос с институциональной почты на shangbin@cs.washington.edu,
с организацией, руководителем при наличии и описанием исследования.
[Официальная папка Google Drive](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs?usp=sharing)
при анонимном HTTP-запросе перенаправляет на вход Google. Подключённого браузера
пользователя с Google-сессией нет. От имени пользователя письма не отправлялись.

[Zenodo](https://zenodo.org/records/7012904) содержит только user.json, label.csv,
split.csv и readme.md. Эти файлы уже скачаны и проверены; они не включают tweets/edges.

## Дополнительно скачанный публичный фрагмент

Источник: [Kaggle / abdelwahedomar/twibot](https://www.kaggle.com/datasets/abdelwahedomar/twibot).
В каталоге API этой поставки один файл tweet_7-004.json. Первые записи имеют id,
author_id, created_at, text и referenced_tweets. Это сторонняя копия, авторство
поставки исходными авторами TwiBot-22 не подтверждено. Это не полный benchmark.

- Архив: **1 781 757 756 байт**.
- JSON после распаковки: **11 923 856 722 байта**.
- Путь: data/raw/twibot22-kaggle-shard7/tweet_7-004.json.
- Архив скачан четырьмя HTTP Range-запросами, размер проверен.
- Архив дополнительно совпал с MD5 публичного облачного объекта; URL/ETag/размер сохранены для resume.
- ZIP CRC и размер JSON проверены при извлечении и повторно на сохранённом файле.
- SHA256 архива: 8bb89c1d9e1980b091c74c69719a48c0a631cd85b0203d2445fbaafaf881cf2e.
- SHA256 JSON: 5dd415d4fc8c6733f557657eaa3d8a3cfd215580786338bd93af766045d36036.
- Подробности: docs/reports/2026-11-19-tweet-shard-download.json.

Найденные другие поставки Kaggle содержали те же профили/метки/split либо
отфильтрованный граф; в публичном каталоге Hugging Face нашлись только TwiBot-20.
Полный набор из девяти tweet-файлов и edge.csv этими проверками не получен.
Графовую копию или TwiBot-20 нельзя незаметно назвать полной поставкой TwiBot-22.

## Подготовленная реальная выборка

Из первых **10 000** публикаций скачанного фрагмента построен отдельный набор:

- 406 авторов: 395 human и 11 bot по официальным меткам.
- Train/val/test: 349 / 34 / 23, исходный split каждого автора сохранён.
- Все 10 000 выбранных публикаций сопоставились с официальными профилями.
- Native author_id числовой; префикс u добавляется только если такой ID действительно
  существует в официальных метках. Reader и регрессионный тест обновлены.
- Raw sample: data/raw/twibot22-shard7-sample/.
- Parquet: data/processed/twibot22-shard7-sample/.
- Полнота истории не объявляется: временные частоты остаются null.

Это **проверка подготовки данных**, не репрезентативный обучающий набор и не
основание публиковать метрики полного benchmark. Выборка взята из начала одного
файла, классы несбалансированы, в val/test мало объектов. Для статистического
эксперимента нужна отдельная процедура отбора и достаточное число классов.

Аудит показал девять общих наблюдаемых признаков текста/URL с JSON-примером Threads:
[отчёт](reports/2026-11-19-shard7-feature-audit.md). Вход Threads здесь синтетический;
реальные поля Meta и качество на Threads этим не подтверждены.

## Повторяемые команды

```bash
# Повторная загрузка/проверка конкретного публичного фрагмента:
.venv/bin/python scripts/download_twibot_shard.py

# Sample/output должны быть пустыми; для повторного отбора указать новые каталоги:
.venv/bin/python -m scripts.prepare_tweet_sample --tweets 10000
.venv/bin/python -m engine.cli validate data/processed/twibot22-shard7-sample
.venv/bin/python -m scripts.audit_features \
  --prepared data/processed/twibot22-shard7-sample \
  --input-kind synthetic_example \
  --output docs/reports/2026-11-19-shard7-feature-audit
```

Загрузчик сохраняет идентичность источника для resume, не соединяет части разных
версий, проверяет кеш и перемещает сомнительные файлы в .unverified/.invalid,
сохраняя их для диагностики. Raw/processed исключены из Git и Docker build context.
Не запускать два загрузчика одновременно для одного каталога.

Для большой подготовки учесть свободное место под архив, JSON и дисковый SQLite
join. /tmp на этой машине — tmpfs; при больших join лучше явно задать TMPDIR на
диске. Текущий reader хранит посты одного аккаунта в памяти, а train_model пока
читает train/val в Python-списки. Большой запуск требует отдельного контроля RAM.

## Готовый запрос полного доступа (не отправлен)

Заполнить реальные сведения, не выдумывать учреждение или руководителя:

```text
To: shangbin@cs.washington.edu
Subject: Research access request for the full TwiBot-22 dataset

Dear Dr. Feng,

I would like to request access to the full TwiBot-22 dataset for research on
language-independent account features and cross-platform bot detection
from Twitter to Threads.

Name: [your name]
Institution: [institution]
Advisor, if applicable: [name / not applicable]
Institutional email: [email]

I have downloaded the public profiles, labels, and official split from Zenodo.
For the behavioral experiments I need the tweet shards and author relations
where required. I will preserve the official splits, cite the dataset and
follow its usage conditions. Could you please provide access instructions?

Thank you,
[your name]
```
