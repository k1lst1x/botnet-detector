# Формат данных

Импорт: страница **Данные** в веб-интерфейсе или

```bash
python manage.py import_csv --accounts data/sample_accounts.csv --posts data/sample_posts.csv --platform twitter
```

## accounts.csv

| колонка | обяз. | описание |
|---|---|---|
| handle | да | логин без `@` |
| display_name | | имя |
| followers, following | | числа |
| created_at | | ISO-8601 / unix / `Wed Oct 10 20:19:24 +0000 2018` |
| label | | `bot` / `human` / `1` / `0` — для обучения |
| true_group | | id кампании (разметка групп для ARI/NMI) |

## posts.csv

| колонка | обяз. | описание |
|---|---|---|
| handle | да | автор |
| post_id | да | уникальный id |
| created_at | да | время публикации |
| text | да | текст (URL, #хэштеги, @упоминания извлекаются автоматически) |
| is_reply, is_repost | | `1/0`, `true/false`; `RT @` в начале → репост |
| reply_to | | handle адресата |

## Kaggle / Cresci-2017 / TwiBot

Приведите колонки к формату выше (обычно `user_id → handle`, `tweet_id → post_id`,
`timestamp → created_at`). Разметку `label` берите из датасета.

**Не переносите качество модели Twitter/X на Threads без проверки** — эксперимент D.

## Threads API

```bash
export THREADS_ACCESS_TOKEN=...   # нужно разрешение threads_keyword_search
python manage.py fetch_threads Алматы Астана тенге --type RECENT --pages 5
```
