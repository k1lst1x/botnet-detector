"""Synthetic Threads-like dataset with controlled coordinated campaigns (slide 04).

Generates humans (irregular, diverse), lone bots (regular, repetitive) and
coordinated campaigns (shared templates, URLs, hashtags, synchronised bursts).
"""
import random
from datetime import datetime, timedelta, timezone

TOPICS = {
    "city": ["Алматы", "Астана", "Шымкент", "пробки", "смог", "метро", "горы", "Медеу", "погода", "снег"],
    "life": ["кофе", "работа", "учёба", "сессия", "универ", "выходные", "семья", "друзья", "кино", "музыка"],
    "tech": ["стартап", "python", "айти", "kaspi", "приложение", "нейросеть", "код", "хакатон", "телефон"],
    "food": ["бешбармак", "баурсаки", "плов", "кафе", "ужин", "завтрак", "чай", "лагман", "доставка"],
    "kk": ["сәлем", "рахмет", "жақсы", "бүгін", "қала", "достар", "керемет", "ауа райы", "жаңалық"],
}
VERBS = ["думаю", "кажется", "люблю", "не понимаю", "обожаю", "устал от", "советую", "пробовал", "жду", "смотрю"]
FILLERS = ["честно", "кстати", "вообще", "короче", "ну", "сегодня", "опять", "наконец-то", "ахах", "))"]

CAMPAIGN_TEMPLATES = [
    ("Новая реформа {x} — это шанс для каждого! Поддержим вместе {tag} {url}", "reform"),
    ("Срочно! {x} меняет всё. Читайте подробности {url} {tag}", "breaking"),
    ("Только честные новости о {x}. Подписывайтесь {tag} {url}", "news"),
    ("Выиграй iPhone! Переходи {url} и участвуй {tag}", "giveaway"),
    ("Инвестиции в {x} дают 40% в месяц, проверено {url} {tag}", "scam"),
]
CAMPAIGN_X = ["экономику", "образование", "тенге", "крипту", "налоги", "ЖКХ", "выборы"]


def _human_text(rng):
    topic = rng.choice(list(TOPICS))
    words = rng.sample(TOPICS[topic], k=min(3, len(TOPICS[topic])))
    parts = [rng.choice(FILLERS), rng.choice(VERBS), *words]
    rng.shuffle(parts)
    text = " ".join(parts)
    if rng.random() < 0.08:
        text += f" #{rng.choice(TOPICS[topic]).replace(' ', '')}"
    if rng.random() < 0.05:
        text += f" https://{rng.choice(['tengrinews.kz', 'youtube.com', 'kaspi.kz', 'instagram.com'])}/{rng.randint(1, 10**6)}"
    return text


def generate(n_humans=220, n_lone_bots=25, n_campaigns=5, campaign_size=(8, 18), days=14, seed=42):
    rng = random.Random(seed)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    accounts, posts = [], []
    pid = 0

    def add_post(handle, ts, text, is_reply=False, is_repost=False, reply_to=""):
        nonlocal pid
        pid += 1
        posts.append({"handle": handle, "post_id": f"syn{pid}", "created_at": ts, "text": text,
                      "is_reply": is_reply, "is_repost": is_repost, "reply_to": reply_to})

    human_handles = [f"user_{i:03d}" for i in range(n_humans)]
    for h in human_handles:
        accounts.append({"handle": h, "display_name": h.replace("_", " ").title(), "label": "human",
                         "followers": rng.randint(20, 3000), "following": rng.randint(50, 900), "true_group": ""})
        rate = rng.uniform(0.3, 6)  # posts/day
        for d in range(days):
            for _ in range(rng.choices(range(12), weights=[max(0.1, 1 - abs(k - rate) / 4) for k in range(12)])[0]):
                hour = rng.choices(range(24), weights=[1, .5, .3, .2, .2, .3, 1, 2, 3, 3, 3, 4, 5, 4, 4, 4, 5, 6, 7, 8, 8, 7, 5, 3])[0]
                ts = start + timedelta(days=d, hours=hour - 5, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
                r = rng.random()
                if r < 0.18:
                    add_post(h, ts, f"@{rng.choice(human_handles)} " + _human_text(rng), is_reply=True,
                             reply_to=rng.choice(human_handles))
                elif r < 0.25:
                    add_post(h, ts, "репост: " + _human_text(rng), is_repost=True)
                else:
                    add_post(h, ts, _human_text(rng))

    # lone bots: very regular, repetitive, many links
    for i in range(n_lone_bots):
        h = f"promo_{i:02d}"
        accounts.append({"handle": h, "display_name": f"Promo {i}", "label": "bot",
                         "followers": rng.randint(0, 80), "following": rng.randint(500, 4000), "true_group": ""})
        interval = rng.uniform(40, 160) * 60
        url = f"https://shop{i}.kz/sale"
        t = start + timedelta(minutes=rng.randint(0, 120))
        phrases = [f"Скидки до 70% только сегодня {url}", f"Лучшие цены в Казахстане {url} #скидки",
                   f"Доставка по всему Казахстану {url}"]
        while t < start + timedelta(days=days):
            add_post(h, t, rng.choice(phrases))
            t += timedelta(seconds=interval * rng.uniform(0.9, 1.1))

    # coordinated campaigns
    for c in range(n_campaigns):
        size = rng.randint(*campaign_size)
        tmpl, kind = CAMPAIGN_TEMPLATES[c % len(CAMPAIGN_TEMPLATES)]
        x = rng.choice(CAMPAIGN_X)
        tag = f"#{kind}{c}kz"
        urls = [f"https://{kind}-{c}.info/{k}" for k in range(3)]
        members = [f"{kind}_{c}_{j:02d}" for j in range(size)]
        for h in members:
            accounts.append({"handle": h, "display_name": f"{x.title()} {rng.randint(1, 99)}", "label": "bot",
                             "followers": rng.randint(5, 300), "following": rng.randint(100, 1500), "true_group": f"campaign_{c}"})
        n_waves = rng.randint(10, 18)
        for _ in range(n_waves):
            wave = start + timedelta(days=rng.uniform(0, days - 0.2))
            url = rng.choice(urls)
            for h in members:
                if rng.random() < 0.85:
                    ts = wave + timedelta(seconds=rng.randint(0, 420))
                    variant = tmpl.format(x=x, tag=tag, url=url)
                    if rng.random() < 0.5:
                        variant = rng.choice(["", "👉 ", "‼️ ", "Друзья, "]) + variant
                    add_post(h, ts, variant)
                if rng.random() < 0.3:  # amplify each other
                    other = rng.choice(members)
                    if other != h:
                        add_post(h, wave + timedelta(seconds=rng.randint(420, 900)),
                                 f"@{other} полностью согласен! {tag}", is_reply=True, reply_to=other)
        # organic filler so they are not trivially separable
        for h in members:
            for _ in range(rng.randint(3, 15)):
                ts = start + timedelta(days=rng.uniform(0, days))
                add_post(h, ts, _human_text(rng))

    # hard negatives: humans reacting to the same real news in a burst
    news_ts = start + timedelta(days=days / 2)
    for h in rng.sample(human_handles, 30):
        add_post(h, news_ts + timedelta(minutes=rng.randint(0, 50)),
                 f"{rng.choice(FILLERS)} землетрясение в Алматы, все целы? #алматы https://tengrinews.kz/news/quake")
    return accounts, posts
