from django.db import models

LABELS = [("bot", "bot"), ("human", "human"), ("unknown", "unknown")]


class EngineJob(models.Model):
    """User-owned reference; job execution and raw data belong to FastAPI."""
    owner = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='engine_jobs')
    job_id = models.CharField(max_length=36, unique=True)
    kind = models.CharField(max_length=32)
    status = models.CharField(max_length=32, default='queued')
    error = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class EngineAccount(models.Model):
    """CRM projection, with analyst labels kept separate from model decisions."""
    owner = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='engine_accounts')
    platform = models.CharField(max_length=16)
    external_user_id = models.CharField(max_length=256)
    username = models.CharField(max_length=256, blank=True)
    label = models.CharField(max_length=16, choices=LABELS, default='unknown')
    notes = models.TextField(blank=True)
    score = models.FloatField(null=True, blank=True)
    decision = models.CharField(max_length=32, blank=True)
    model_version = models.CharField(max_length=128, blank=True)
    source_job = models.ForeignKey(EngineJob, null=True, blank=True, on_delete=models.SET_NULL)
    queue = models.CharField(max_length=16, blank=True)
    group = models.ForeignKey('EngineGroup', null=True, blank=True, on_delete=models.SET_NULL, related_name='accounts')
    coordination_score = models.FloatField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        constraints = [models.UniqueConstraint(fields=['owner', 'platform', 'external_user_id'], name='crm_owner_platform_external')]


class EngineGroup(models.Model):
    """Coordinated group found by one analysis job (co-activity network community)."""
    owner = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='engine_groups')
    source_job = models.ForeignKey(EngineJob, on_delete=models.CASCADE, related_name='groups')
    group_id = models.CharField(max_length=32)
    size = models.PositiveIntegerField()
    density = models.FloatField()
    mean_weight = models.FloatField()
    flagged = models.BooleanField(default=False)
    members = models.JSONField(default=list)
    shared_objects = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-flagged', '-size', '-mean_weight']
        constraints = [models.UniqueConstraint(fields=['source_job', 'group_id'], name='crm_job_group')]
