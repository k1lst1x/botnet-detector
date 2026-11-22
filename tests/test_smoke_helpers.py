import pytest

from scripts.smoke_mvp import label_form_for_author


def test_smoke_selects_only_its_own_account_even_if_other_account_is_first():
    html = """<section class="card"><h2>threads · real-person</h2><form action="/engine/accounts/1/"></form></section>
    <section class="card"><h2>threads · smoke-unique</h2><form action="/engine/accounts/2/"></form></section>"""
    assert label_form_for_author(html, "smoke-unique") == "/engine/accounts/2/"
    with pytest.raises(ValueError):
        label_form_for_author(html, "absent-author")
