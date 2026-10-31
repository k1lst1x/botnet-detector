from django.db import models


class AnalysisRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    params = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    model_kind = models.CharField(max_length=64, blank=True)
    n_accounts = models.IntegerField(default=0)
    n_edges = models.IntegerField(default=0)
    n_communities = models.IntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Run #{self.pk} ({self.created_at:%Y-%m-%d %H:%M})"


class Community(models.Model):
    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="communities")
    method = models.CharField(max_length=16, default="louvain")
    label = models.IntegerField()
    size = models.IntegerField(default=0)
    density = models.FloatField(default=0)
    mean_weight = models.FloatField(default=0)
    mean_bot_proba = models.FloatField(default=0)
    coordination_score = models.FloatField(default=0)
    top_signals = models.JSONField(default=dict)
    shared_urls = models.JSONField(default=list)
    shared_hashtags = models.JSONField(default=list)

    class Meta:
        ordering = ["-coordination_score"]

    def __str__(self):
        return f"{self.method}#{self.label} ({self.size})"


class Account(models.Model):
    LABELS = [("bot", "bot"), ("human", "human"), ("unknown", "unknown")]

    handle = models.CharField(max_length=128, unique=True)
    display_name = models.CharField(max_length=256, blank=True)
    platform = models.CharField(max_length=32, default="threads")
    created_at = models.DateTimeField(null=True, blank=True)
    followers = models.IntegerField(default=0)
    following = models.IntegerField(default=0)
    label = models.CharField(max_length=16, choices=LABELS, default="unknown")
    true_group = models.CharField(max_length=64, blank=True, help_text="ground-truth campaign id (synthetic / labelled)")

    # analysis output
    features = models.JSONField(default=dict, blank=True)
    explanation = models.JSONField(default=list, blank=True)
    bot_proba = models.FloatField(null=True, blank=True)
    anomaly_score = models.FloatField(null=True, blank=True)
    risk = models.FloatField(null=True, blank=True, db_index=True)
    community = models.ForeignKey(Community, null=True, blank=True, on_delete=models.SET_NULL, related_name="members")
    dbscan_cluster = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-risk", "handle"]

    def __str__(self):
        return f"@{self.handle}"

    @property
    def risk_level(self):
        if self.risk is None:
            return "—"
        if self.risk >= 0.7:
            return "high"
        if self.risk >= 0.4:
            return "medium"
        return "low"


class Post(models.Model):
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name="posts")
    external_id = models.CharField(max_length=128, unique=True)
    text = models.TextField(blank=True)
    created_at = models.DateTimeField(db_index=True)
    is_reply = models.BooleanField(default=False)
    is_repost = models.BooleanField(default=False)
    reply_to = models.CharField(max_length=128, blank=True, help_text="handle of the addressee")
    urls = models.JSONField(default=list, blank=True)
    hashtags = models.JSONField(default=list, blank=True)
    mentions = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-created_at"]


class Edge(models.Model):
    """Weighted coordination edge W(A,B)."""

    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="edges")
    a = models.ForeignKey(Account, on_delete=models.CASCADE, related_name="+")
    b = models.ForeignKey(Account, on_delete=models.CASCADE, related_name="+")
    weight = models.FloatField()
    components = models.JSONField(default=dict)
