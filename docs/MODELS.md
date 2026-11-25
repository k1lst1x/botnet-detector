# Файлы моделей

Обученные модели не хранятся в Git. Единственное место обмена — папка Google Drive:
**https://drive.google.com/drive/folders/1JLmfTDU9aPYOnZ49P_aW9yg1Tjy9DILI**

- Модели **скачивать отсюда**.
- Обновления моделей **загружать сюда же** (нужно право редактора папки).

## Состав

| Папка | Что это | SHA256 `model.joblib` |
|---|---|---|
| `review-profile-v2/` | профильная модель, TwiBot-22, 700 тыс. профилей; для очереди проверки | `0407ee4cb98c37db81a96a7bbf6d4fdd2fa83fb0a241b35c4a634ccf25ab2a0f` |
| `cresci-account-v1/` | профиль + поведение, Cresci-2017, 8 541 аккаунт с ≥20 твитами; исследовательская, на Threads не проверена | `3085d7dcf4cdcfe428771963227a5b01812a5e3997f3d4b37a541012361f281c` |

В каждой папке рядом лежит `report.json` с метриками. `cresci-account-v1` требует полной истории
и от 20 публикаций, иначе система вернёт `insufficient_data`.

## Скачать и подключить

1. Скачать папку модели в `media/models/<имя>/`.
2. Сверить контрольную сумму с таблицей выше **до запуска**: `joblib` при загрузке может выполнить
   произвольный код, поэтому файл с другой суммой не использовать.

   ```bash
   sha256sum media/models/review-profile-v2/model.joblib
   ```
3. Сделать модель активной и перезапустить процессы. Прежний `portable.joblib` сохранить отдельно.

   ```bash
   cp media/models/review-profile-v2/model.joblib media/models/portable.joblib
   docker compose restart api worker
   ```

API и worker читают `ENGINE_MODEL_PATH`; в Compose это `/app/media/models/portable.joblib`.

## Обновить модель

1. Собрать новую версию **в новую папку** (`<имя>-vN`): скрипты сборки не перезаписывают существующие.
2. Загрузить `model.joblib` и `report.json` в одноимённую папку на Drive. Старые версии не удалять и не заменять.
3. Добавить строку с SHA256 в таблицу этого файла и провести изменение через pull request.

Сборка: `scripts.build_profile_v2_model` (review-profile-v2), `scripts.build_cresci_model` (cresci-account-v1).
