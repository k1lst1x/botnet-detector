from django.urls import path

from . import views

urlpatterns = [
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
