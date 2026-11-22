from django.contrib import admin

from .models import EngineAccount, EngineJob


@admin.register(EngineJob)
class EngineJobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'owner', 'kind', 'status', 'created_at')
    list_filter = ('kind', 'status')
    readonly_fields = ('job_id', 'owner', 'kind', 'status', 'error', 'created_at')


@admin.register(EngineAccount)
class EngineAccountAdmin(admin.ModelAdmin):
    list_display = ('username', 'platform', 'owner', 'label', 'decision', 'score')
    list_filter = ('platform', 'label', 'decision')
    search_fields = ('username', 'external_user_id')
