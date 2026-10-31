from django import template

register = template.Library()


@register.filter
def pct(v):
    return "—" if v is None else f"{v * 100:.0f}%"


@register.filter
def num(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    return v


@register.filter
def get(d, k):
    return (d or {}).get(k)


@register.filter
def level(v):
    if v is None:
        return "none"
    return "high" if v >= 0.7 else "medium" if v >= 0.4 else "low"
