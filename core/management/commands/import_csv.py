from django.core.management.base import BaseCommand

from core.importer import import_records, read_csv


class Command(BaseCommand):
    help = "Импорт accounts.csv / posts.csv (Kaggle Twitter, Cresci, TwiBot после конвертации, Threads export)"

    def add_arguments(self, parser):
        parser.add_argument("--accounts", help="путь к accounts.csv")
        parser.add_argument("--posts", help="путь к posts.csv")
        parser.add_argument("--platform", default="twitter")

    def handle(self, *a, **o):
        accounts = read_csv(open(o["accounts"], "rb")) if o.get("accounts") else []
        posts = read_csv(open(o["posts"], "rb")) if o.get("posts") else []
        na, np_ = import_records(accounts, posts, platform=o["platform"])
        self.stdout.write(self.style.SUCCESS(f"Аккаунтов: {na}, постов: {np_}"))
