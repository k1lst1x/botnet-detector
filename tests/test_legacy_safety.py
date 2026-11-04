from unittest.mock import patch

import pytest

from analysis.pipeline import run_analysis


@pytest.mark.django_db
def test_legacy_analysis_without_model_does_not_implicitly_fit():
    from core.models import Account

    Account.objects.create(handle="safety")
    with (
        patch("analysis.pipeline.ml.load", return_value=None),
        patch(
            "analysis.pipeline.train_and_evaluate",
            side_effect=AssertionError("implicit training"),
        ),
    ):
        with pytest.raises(RuntimeError, match="model_not_ready"):
            run_analysis()
