import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from analysis.twibot import TwiBotFormatError, load_dataset, save_artifacts, train_baseline


class Command(BaseCommand):
    help = "Обучить языконезависимый profile baseline на официальных split TwiBot-22"

    def add_arguments(self, parser):
        parser.add_argument("--dataset", required=True, help="папка с user.json, label.csv и split.csv")
        parser.add_argument("--output", default="media/models/twibot22-baseline")
        parser.add_argument("--model", default="auto", choices=["auto", "xgboost", "sklearn"])
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max-per-split", type=int, help="лимит для локального smoke-test; не для итоговых метрик")

    def handle(self, *args, **options):
        try:
            rows = load_dataset(Path(options["dataset"]), options["max_per_split"])
            bundle, metrics = train_baseline(rows, options["model"], options["seed"])
            if options["max_per_split"]:
                metrics["warning"] = "sampled smoke-test; do not report these metrics as final"
            save_artifacts(bundle, metrics, Path(options["output"]))
        except TwiBotFormatError as error:
            raise CommandError(str(error)) from error
        self.stdout.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        self.stdout.write(self.style.SUCCESS(f"Артефакты: {options['output']}"))
