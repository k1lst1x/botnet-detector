import builtins
from unittest.mock import patch

from django.test import Client, TestCase

from analysis.graph import time_sim
from analysis.ml import make_classifier
from analysis.pipeline import run_analysis
from analysis.synthetic import generate
from analysis.text import extract_entities, jaccard
from core.importer import import_records
from core.models import Account, Community


class UnitTests(TestCase):
    def test_auto_model_falls_back_when_xgboost_cannot_load(self):
        original_import = builtins.__import__

        def unavailable_optional_model(name, *args, **kwargs):
            if name == "xgboost":
                raise OSError("native XGBoost library is unavailable")
            if name == "catboost":
                raise ImportError("CatBoost is not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=unavailable_optional_model):
            name, _ = make_classifier("auto")
        self.assertEqual(name, "sklearn-gb")

    def test_entities(self):
        urls, tags, mentions = extract_entities("Читай https://www.Site.kz/a?utm=1 #Алматы @User.One")
        self.assertEqual(urls, ["site.kz/a"])
        self.assertEqual(tags, ["алматы"])
        self.assertEqual(mentions, ["user.one"])

    def test_jaccard(self):
        self.assertEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard(set(), {"a"}), 0.0)

    def test_time_sim_neighbour_window(self):
        self.assertEqual(time_sim({1, 2}, {3, 100}), 0.5)  # bucket 2 matches 3 (±1 window)


class PipelineTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        a, p = generate(n_humans=80, n_lone_bots=8, n_campaigns=3, campaign_size=(8, 10), days=7, seed=1)
        import_records(a, p, platform="synthetic")

    def test_end_to_end_detects_campaigns(self):
        run = run_analysis(retrain=True)
        self.assertGreaterEqual(run.n_communities, 3)
        self.assertIn("B_extended", run.metrics)
        self.assertGreater(run.metrics["C_groups"]["louvain_ARI"], 0.6)
        top = Community.objects.filter(run=run).first()
        bots = top.members.filter(label="bot").count()
        self.assertGreater(bots / top.size, 0.8)
        self.assertTrue(Account.objects.exclude(risk=None).exists())

    def test_pages_render(self):
        run_analysis(retrain=True)
        c = Client()
        acc = Account.objects.first()
        for url in ["/", "/accounts/", f"/accounts/{acc.handle}/", "/communities/", "/graph/", "/data/",
                    "/api/graph/", "/api/accounts/", "/api/communities/"]:
            self.assertEqual(c.get(url).status_code, 200, url)
