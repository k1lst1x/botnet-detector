from django.urls import path
from django.views.generic import RedirectView

from . import engine_views

urlpatterns = [
    path("engine/", engine_views.jobs, name="engine_jobs"),
    path("engine/import/", engine_views.import_file, name="engine_import"),
    path("engine/collect/", engine_views.collect, name="engine_collect"),
    path("engine/jobs/<str:job_id>/", engine_views.detail, name="engine_job"),
    path("engine/jobs/<str:job_id>/status/", engine_views.status, name="engine_status"),
    path("engine/jobs/<str:job_id>/refresh/", engine_views.refresh, name="engine_refresh"),
    path("engine/jobs/<str:job_id>/analyze/", engine_views.analyze, name="engine_analyze"),
    path("engine/accounts/", engine_views.accounts, name="engine_accounts"),
    path("engine/accounts/<int:pk>/", engine_views.label, name="engine_label"),
    path("engine/groups/", engine_views.groups, name="engine_groups"),
    path("engine/groups/<int:pk>/", engine_views.group_detail, name="engine_group"),
    path("", RedirectView.as_view(pattern_name="engine_jobs", permanent=False), name="home"),
]
