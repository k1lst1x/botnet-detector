from django.contrib import admin

from .models import Account, AnalysisRun, Community, Edge, Post


@admin.register(Account)
class AccountAdmin(admin.ModelAdmin):
    list_display = ("handle", "platform", "label", "bot_proba", "anomaly_score", "risk", "community")
    list_filter = ("platform", "label")
    search_fields = ("handle", "display_name")


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ("account", "created_at", "is_reply", "is_repost", "text")
    search_fields = ("text", "account__handle")
    raw_id_fields = ("account",)


@admin.register(Community)
class CommunityAdmin(admin.ModelAdmin):
    list_display = ("run", "method", "label", "size", "density", "coordination_score")


admin.site.register(AnalysisRun)
admin.site.register(Edge)
