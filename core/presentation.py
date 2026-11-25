"""Plain-language view of engine results: statuses, decisions, features, account cards."""

import math

JOB_KINDS = {"import": "Импорт файла", "threads": "Поиск в Threads", "analysis": "Проверка аккаунтов"}
JOB_STATUSES = {"queued": "В очереди", "running": "Выполняется", "completed": "Готово", "failed": "Ошибка"}
JOB_ERRORS = {
    "invalid_input": "Файл не удалось разобрать. Проверьте формат, обязательные поля и даты.",
    "internal_error": "Внутренняя ошибка обработки. Попробуйте запустить ещё раз.",
    "threads_not_configured": "Доступ к Threads API не настроен. Используйте импорт файла.",
    "token_expired": "Срок действия токена Threads истёк.",
    "permission_denied": "У токена Threads нет прав на поиск.",
    "upstream_unavailable": "Threads сейчас недоступен. Повторите позже.",
    "upstream_rejected": "Threads отклонил запрос.",
}
DECISIONS = {
    "priority_review": "Похож на автоматизированный аккаунт — проверить в первую очередь",
    "standard_review": "По самому аккаунту явных признаков автоматизации не найдено",
    "needs_review": "Нужна ручная проверка",
    "insufficient_data": "Недостаточно данных для оценки",
    "model_not_ready": "Оценки нет: модель ещё не обучена или не была установлена",
    "model_incompatible": "Оценки нет: файл модели не подходит к текущим признакам",
    "bot": "Модель относит аккаунт к ботам",
    "human": "Модель относит аккаунт к людям",
}
# name: (label, value kind)
FEATURES = {
    "followers_log1p": ("Подписчики", "log_count"),
    "following_log1p": ("Подписки", "log_count"),
    "followers_following_ratio": ("Подписчиков на одну подписку", "number"),
    "bio_present": ("Описание профиля заполнено", "bool"),
    "avatar_present": ("Есть аватар", "bool"),
    "avatar_default": ("Стандартный аватар", "bool"),
    "account_age_days": ("Возраст аккаунта, дней", "count"),
    "posts_per_day": ("Публикаций в день", "number"),
    "interval_median_s": ("Обычный перерыв между публикациями", "seconds"),
    "interval_p90_s": ("Длинный перерыв между публикациями (90 %)", "seconds"),
    "interval_cv": ("Неравномерность перерывов", "number"),
    "burstiness": ("Всплески активности (от −1 до 1)", "number"),
    "short_interval_ratio": ("Доля публикаций почти подряд", "percent"),
    "hour_entropy": ("Разброс по часам суток (0–1)", "number"),
    "circadian_amplitude": ("Выраженность суточного ритма (0–1)", "number"),
    "max_posts_per_hour": ("Максимум публикаций за час", "count"),
    "active_day_ratio": ("Доля дней с публикациями", "percent"),
    "reply_ratio": ("Доля ответов", "percent"),
    "repost_ratio": ("Доля репостов", "percent"),
    "quote_ratio": ("Доля цитат", "percent"),
    "media_ratio": ("Доля публикаций с медиа", "percent"),
    "duplicate_ratio": ("Доля повторяющихся текстов", "percent"),
    "text_length_median": ("Обычная длина текста, символов", "count"),
    "text_length_cv": ("Разброс длины текста", "number"),
    "text_digit_ratio": ("Доля цифр в тексте", "percent"),
    "text_space_ratio": ("Доля пробелов в тексте", "percent"),
    "text_punctuation_ratio": ("Доля знаков препинания", "percent"),
    "url_ratio": ("Доля публикаций со ссылками", "percent"),
    "unique_domains": ("Разных сайтов в ссылках", "count"),
    "domain_concentration": ("Сосредоточенность ссылок на одном сайте", "percent"),
    "unique_targets": ("Разных адресатов", "count"),
    "target_concentration": ("Сосредоточенность на одних адресатах", "percent"),
    "bio_length": ("Длина описания, символов", "count"),
    "bio_url_count": ("Ссылок в описании", "count"),
    "bio_mention_count": ("Упоминаний в описании", "count"),
    "bio_hashtag_count": ("Хэштегов в описании", "count"),
    "bio_digit_ratio": ("Доля цифр в описании", "percent"),
    "name_length": ("Длина имени, символов", "count"),
    "name_digit_ratio": ("Доля цифр в имени", "percent"),
    "name_non_alnum_ratio": ("Доля спецсимволов в имени", "percent"),
    "username_length": ("Длина ника, символов", "count"),
    "username_digit_ratio": ("Доля цифр в нике", "percent"),
    "username_trailing_digits": ("Цифр в конце ника", "count"),
    "username_underscore_count": ("Подчёркиваний в нике", "count"),
    "name_username_overlap": ("Совпадение имени и ника", "percent"),
}


def _number(value):
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",") or "0"


def _count(value):
    return f"{int(round(value)):,}".replace(",", " ")


def _duration(seconds):
    if seconds < 60:
        return f"{int(seconds)} с"
    if seconds < 3600:
        return f"{int(seconds // 60)} мин"
    if seconds < 172800:
        return f"{_number(round(seconds / 3600, 1))} ч"
    return f"{_number(round(seconds / 86400, 1))} дн."


def feature_value(name, value):
    if value is None:
        return "нет данных"
    kind = FEATURES.get(name, ("", "number"))[1]
    if kind == "bool":
        return "да" if value else "нет"
    if kind == "percent":
        return f"{_number(round(value * 100, 1))} %"
    if kind == "seconds":
        return _duration(value)
    if kind == "log_count":
        return _count(math.expm1(value))
    return _count(value) if kind == "count" else _number(value)


def feature_rows(features):
    """Known features only; technical missing-value masks are hidden."""
    return [{"label": FEATURES[n][0], "value": feature_value(n, v)}
            for n, v in (features or {}).items() if n in FEATURES and v is not None]


def reasons(explanation):
    out = []
    for item in explanation or []:
        name = item["feature"]
        missing = name.endswith("__missing")
        base = name.removesuffix("__missing")
        impact = item["contribution_log_odds"]
        if not impact or base not in FEATURES:
            continue
        out.append({
            "label": FEATURES[base][0],
            "value": ("данных нет" if item.get("value") else "данные есть") if missing
            else feature_value(base, item.get("value")),
            "up": impact > 0,
        })
    return out


def details_from_row(row):
    """The part of one engine result kept in CRM for the account card."""
    return {
        "profile": row.get("profile") or {},
        "activity": row.get("activity") or {},
        "coverage": row.get("coverage") or {},
        "features": {k: v for k, v in (row.get("features") or {}).items() if k in FEATURES},
        "explanation": row.get("explanation") or [],
        "threshold": row.get("threshold"),
        "warnings": row.get("warnings") or [],
    }


def account_card(details, *, platform, external_user_id, username, decision, score, queue):
    details = details or {}
    profile, features = details.get("profile") or {}, details.get("features") or {}
    coverage, activity = details.get("coverage") or {}, details.get("activity") or {}
    facts = []

    def followers(key, feature):
        if profile.get(key) is not None:
            return _count(profile[key])
        return feature_value(feature, features[feature]) if features.get(feature) is not None else None

    def fact(label, value):
        if value not in (None, ""):
            facts.append({"label": label, "value": value})

    fact("Подписчики", followers("followers", "followers_log1p"))
    fact("Подписки", followers("following", "following_log1p"))
    if coverage.get("posts") is not None:
        fact("Публикаций в данных", _count(coverage["posts"]))
    for name in ("posts_per_day", "interval_median_s", "url_ratio", "duplicate_ratio", "account_age_days"):
        if features.get(name) is not None:
            fact(FEATURES[name][0], feature_value(name, features[name]))
    if profile.get("is_verified") is not None:
        fact("Подтверждён", "да" if profile["is_verified"] else "нет")
    if activity.get("first") and activity.get("last"):
        a, b = (".".join(reversed(activity[k][:10].split("-"))) for k in ("first", "last"))
        fact("Период публикаций", a if a == b else f"{a} – {b}")
    tone = {"priority": "bot", "standard": "human"}.get(queue, "")
    if not queue:
        tone = {"priority_review": "bot", "standard_review": "human"}.get(decision, "")
    threshold = details.get("threshold")
    return {
        "title": profile.get("display_name") or username or external_user_id,
        "handle": username or external_user_id,
        "platform": {"threads": "Threads", "twitter": "Twitter"}.get(platform, platform),
        "bio": profile.get("biography"),
        "decision": DECISIONS.get(decision, decision or "Ещё не проверялся"),
        "queue": {"priority": "приоритетная проверка", "standard": "обычная очередь",
                  "not_scored": "без оценки"}.get(queue, ""),
        "tone": tone,
        "meter_tone": "bot" if decision in ("priority_review", "bot") else "human",
        "score": None if score is None else _number(score),
        "score_percent": None if score is None else round(score * 100),
        "threshold": None if threshold is None else _number(threshold),
        "threshold_percent": None if threshold is None else round(threshold * 100),
        "facts": facts,
        "reasons": reasons(details.get("explanation")),
        "features": feature_rows(features),
        "posts": activity.get("recent") or [],
        "incomplete": coverage.get("complete") is False,
    }
