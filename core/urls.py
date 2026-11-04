from django.urls import path

from . import views, engine_views

urlpatterns = [
    path("engine/", engine_views.jobs, name="engine_jobs"),
    path("engine/import/", engine_views.import_file, name="engine_import"),
    path("engine/collect/", engine_views.collect, name="engine_collect"),
    path("engine/jobs/<str:job_id>/", engine_views.detail, name="engine_job"),
    path("engine/jobs/<str:job_id>/refresh/", engine_views.refresh, name="engine_refresh"),
    path("engine/jobs/<str:job_id>/analyze/", engine_views.analyze, name="engine_analyze"),
    path("engine/accounts/", engine_views.accounts, name="engine_accounts"),
    path("engine/accounts/<int:pk>/", engine_views.label, name="engine_label"),
    path("", views.dashboard, name="dashboard"),
    path("accounts/", views.accounts, name="accounts"),
    path("accounts/<str:handle>/", views.account_detail, name="account"),
    path("communities/", views.communities, name="communities"),
    path("communities/<int:pk>/", views.community_detail, name="community"),
    path("graph/", views.graph, name="graph"),
    path("data/", views.data, name="data"),
    path("data/upload/", views.upload, name="upload"),
    path("data/synthetic/", views.synthetic, name="synthetic"),
    path("data/reset/", views.reset, name="reset"),
    path("analyze/", views.analyze, name="analyze"),
    path("api/accounts/", views.api_accounts, name="api_accounts"),
    path("api/accounts/<str:handle>/", views.api_account, name="api_account"),
    path("api/communities/", views.api_communities, name="api_communities"),
    path("api/graph/", views.api_graph, name="api_graph"),
]
