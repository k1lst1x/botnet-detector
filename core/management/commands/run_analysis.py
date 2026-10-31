import json

from django.core.management.base import BaseCommand

from analysis.graph import GraphParams
from analysis.pipeline import run_analysis


class Command(BaseCommand):
    help = "Полный pipeline: признаки → ML → граф координации → сообщества → риск"

    def add_arguments(self, parser):
        parser.add_argument("--retrain", action="store_true", help="переобучить модель на размеченных аккаунтах")
        parser.add_argument("--model", default="auto", choices=["auto", "xgboost", "catboost", "sklearn"])
        parser.add_argument("--threshold", type=float, default=0.30)
        parser.add_argument("--window", type=int, default=600, help="Δt в секундах")

    def handle(self, *a, **o):
        params = GraphParams(threshold=o["threshold"], window_s=o["window"])
        run = run_analysis(params, retrain=o["retrain"], model_kind=o["model"])
        self.stdout.write(self.style.SUCCESS(
            f"{run}: аккаунтов={run.n_accounts}, рёбер={run.n_edges}, сообществ={run.n_communities}, модель={run.model_kind}"))
        self.stdout.write(json.dumps(run.metrics, ensure_ascii=False, indent=2))
