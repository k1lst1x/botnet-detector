# Первый структурный эксперимент

Запускать из корня `botnet-detector`, используя `.venv`. Конфигурация фиксируется до обучения в `experiments/structural-shard7-v1.json`.

```sh
.venv/bin/python -m scripts.run_training_experiment train
.venv/bin/python -m scripts.run_training_experiment evaluate
```

Первый этап читает только train и val. Он проверяет SHA256 подготовленных матриц, обучает imputer и XGBoost только на train, сравнивает с постоянной вероятностью train prior. На validation выбираются лучший F1 и максимальный recall при precision ≥0.9 и минимум 100 положительных прогнозах. Недостижимая цель записывается как null. Перебора параметров и балансировки классов нет.

В `media/models/experiments/structural-shard7-v1-run1/` сохраняются план, модель, validation и frozen.json с хешами и версиями библиотек. Второй этап проверяет неизменность файлов, затем впервые читает test и применяет уже выбранные пороги. Повторный запуск в ту же папку запрещён. Test нельзя использовать для последующей подстройки этого эксперимента.

Артефакт совместим с ModelService, но помечен `decision_policy=review_only`: оценка не означает автоматического обвинения в ботоводстве. Нужны минимум 20 постов; применяется тот же bottom-100 BLAKE2b sampling с seed 42. Рабочий `media/models/portable.joblib` не заменяется.

Ограничения: единственный сторонний фрагмент TwiBot-22, не полный benchmark; дисбаланс доли ботов между train/val/test; текстовые структурные признаки могут зависеть от платформы и письменности. Языковая лексика не используется, но языконезависимость качества пока не доказана. Проверки на размеченных Threads нет, score не откалиброван.

## Второй запуск (20 ноября 2026)

[Аудит и weighted-v2](reports/2026-11-20-weighted-validation.md): веса классов не дали существенного улучшения. Новый config `experiments/structural-shard7-weighted-v2.json` использует прежние параметры и `scale_pos_weight=21.732908201097434`, вычисленный только по train. Политика `validation_only` запрещает test-evaluate. Отдельный аудит: `.venv/bin/python -m scripts.audit_training_shift`; выходной JSON не перезаписывается.
