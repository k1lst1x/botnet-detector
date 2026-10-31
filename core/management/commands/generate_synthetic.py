from django.core.management.base import BaseCommand

from analysis.synthetic import generate
from core.importer import import_records
from core.models import Account


class Command(BaseCommand):
    help = "Сгенерировать синтетический датасет с координированными кампаниями"

    def add_arguments(self, parser):
        parser.add_argument("--humans", type=int, default=220)
        parser.add_argument("--lone-bots", type=int, default=25)
        parser.add_argument("--campaigns", type=int, default=5)
        parser.add_argument("--days", type=int, default=14)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--reset", action="store_true", help="удалить существующие данные")

    def handle(self, *a, **o):
        if o["reset"]:
            Account.objects.all().delete()
        accounts, posts = generate(o["humans"], o["lone_bots"], o["campaigns"], days=o["days"], seed=o["seed"])
        na, np_ = import_records(accounts, posts, platform="synthetic")
        self.stdout.write(self.style.SUCCESS(f"Импортировано аккаунтов: {na}, постов: {np_}"))
