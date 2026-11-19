# Запуск MVP: PostgreSQL + Redis + Django + FastAPI

Дата: 2026-11-19, Asia/Almaty.

## Результат

Локальный стек запущен на http://127.0.0.1:8100/engine/ и http://127.0.0.1:8101/docs.
Порт 8001 занят другим проектом; его контейнеры не изменялись. В Compose добавлены
DJANGO_PORT/ENGINE_PORT с прежними значениями по умолчанию; текущие 8100/8101
сохранены только в локальном .env. Сервисы оставлены работающими на localhost.

Проверены оба init/migrate: PostgreSQL botnet_engine и botnet_crm, Redis, API,
worker и Django. Реальные HTTP-формы использовали CSRF и сессии, задания прошли
через реальный Redis и Celery, данные сохранились в реальной PostgreSQL.
Это HTTP-проверка отрисованных форм, а не визуальное тестирование браузером.

## Пройденные сценарии

- live_api_and_explicit_missing_model
- real_django_login_and_anonymous_redirect
- web_import_and_idempotent_repeat
- postgres_redis_worker_completed_import
- real_analysis_features_and_missing_model_result
- django_and_fastapi_tenant_isolation
- manual_crm_label_survives_result_refresh
- completed_sql_jobs_and_crm_survive_worker_restart
- real_redis_queued_job_consumed_after_worker_restart
- broker_outage_preserves_visible_sql_job_in_django
- recover_republishes_persisted_job_without_duplicate_submission

- PostgreSQL: устаревшая попытка worker не может завершить новую попытку (lease token).

Первый скрипт создаёт изолированные fixture-задания в аккаунте локального администратора.
Второй временно останавливает только worker и Redis этого проекта и восстанавливает
их в finally. Другие Compose-проекты и volumes не затрагиваются. Данные smoke-заданий
оставлены для просмотра в CRM.

## Воспроизведение

```bash
# Если окружение ещё не запущено: подготовить .env, создать media до Docker.
mkdir -p media
docker compose up --build -d
# Только локальная разработка при DEBUG=1; существующие пароли не сбрасываются:
docker compose exec -T web python manage.py shell < scripts/provision_dev_users.py
.venv/bin/python scripts/smoke_mvp.py
# Временно останавливает локальные сервисы проекта:
.venv/bin/python scripts/smoke_recovery.py
.venv/bin/python -m scripts.audit_features --input-kind synthetic_example
```

Файл media/dev-login.json содержит данные локального входа, имеет права 0600,
исключён из Git и Docker build context. Пароли и API key в этот отчёт не включены.
Не запускать provision_dev_users против production-окружения.
Подробные JSON-результаты: media/mvp-smoke-report.json и media/mvp-recovery-report.json.

## Проверки кода

До запуска инфраструктуры: pytest — 38 passed, 2 skipped. Два пропуска — старые
тесты, запускающие обучение. Fit не выполнялся. Ruff проверяет новые скрипты;
Compose config валиден. Финальный прогон после изменений: **44 passed, 2 skipped**. Django check, migration
drift, Ruff и Compose config прошли. Обновлённый стенд повторно прошёл smoke и recovery.

## Ограничения и следующий этап

Модель не обучена: результат model_not_ready с null score является ожидаемым.
Проверен рабочий путь без модели, не готовность классификатора ботов.
Реального экспорта Threads и Meta credentials нет. Аудит feature-audit.md использует
явно обозначенный синтетический пример и не подтверждает доступность live API.
Нужны публикации Twitter и реальная выборка Threads для выбора совместимого набора.
