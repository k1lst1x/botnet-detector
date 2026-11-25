from django import template

register = template.Library()

KINDS = {
    "m": "адресат",
    "u": "ссылка",
    "h": "хэштег",
    "txt": "одинаковый текст",
    "rt": "ретвит поста",
    "rtu": "ретвит автора",
    "re": "ответ на пост",
}


@register.filter
def object_kind(token):
    kind, _, _ = str(token).partition(":")
    return KINDS.get(kind, kind)


@register.filter
def object_value(token):
    kind, _, value = str(token).partition(":")
    if kind == "m":
        return value if value.startswith("id:") else f"@{value}"
    if kind == "h":
        return f"#{value}"
    if kind == "txt":
        return "тексты совпадают без учёта ссылок, упоминаний и цифр"
    return value


@register.filter
def member_handle(key):
    """'threads:123' -> '123'."""
    return str(key).partition(":")[2] or key
