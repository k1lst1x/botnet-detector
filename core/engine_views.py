import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Case, F, When
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .engine_client import EngineClient, EngineClientError
from .models import EngineAccount, EngineGroup, EngineJob
from .presentation import account_card, details_from_row


POLLED_JOBS = 10
SHOWN_RESULTS = 200


def _remember(owner, result, kind, title="", source=None):
    job, _ = EngineJob.objects.get_or_create(
        job_id=result["job_id"],
        owner=owner,
        defaults={"kind": kind, "status": result["status"], "title": title[:200], "source": source},
    )
    return job


@login_required
def jobs(request):
    listed = list(EngineJob.objects.filter(owner=request.user).select_related("source")[:100])
    client = EngineClient(request.user)
    try:
        for job in [j for j in listed if j.is_active][:POLLED_JOBS]:
            _sync(request.user, job, client)
    except EngineClientError as exc:
        messages.error(request, str(exc))
    return render(
        request,
        "core/engine_jobs.html",
        {"jobs": listed, "submission_key": str(uuid.uuid4())},
    )


@login_required
@require_POST
def import_file(request):
    file = request.FILES.get("file")
    if not file or file.size > 8_000_000:
        messages.error(request, "Выберите файл до 8 МБ")
        return redirect("engine_jobs")
    try:
        content = file.read().decode("utf-8-sig")
        result = EngineClient(request.user).submit(
            "/v1/imports",
            {"format": request.POST.get("format", "csv"), "content": content},
            request.POST.get("idempotency_key"),
        )
        job = _remember(request.user, result, "import", file.name)
        if result.get("dispatch_pending"):
            messages.warning(
                request,
                "Задание сохранено. Очередь временно недоступна; администратор сможет восстановить запуск.",
            )
        return redirect("engine_job", job_id=job.job_id)
    except UnicodeError:
        messages.error(request, "Нужен файл в UTF-8")
    except EngineClientError as exc:
        messages.error(request, str(exc))
    return redirect("engine_jobs")


@login_required
@require_POST
def collect(request):
    query = request.POST.get("query", "").strip()
    if not query:
        messages.error(request, "Укажите поисковый запрос")
        return redirect("engine_jobs")
    try:
        result = EngineClient(request.user).submit(
            "/v1/collections/threads",
            {"query": query, "max_pages": 5},
            request.POST.get("idempotency_key"),
        )
        job = _remember(request.user, result, "threads", query)
        if result.get("dispatch_pending"):
            messages.warning(
                request,
                "Задание сохранено. Очередь временно недоступна; администратор сможет восстановить запуск.",
            )
        return redirect("engine_job", job_id=job.job_id)
    except EngineClientError as exc:
        messages.error(request, str(exc))
    return redirect("engine_jobs")


def _save_groups(owner, job, summary):
    """Store the job's groups, keeping existing rows so their page links stay valid."""
    saved = {}
    with transaction.atomic():
        for g in summary.get("groups", []):
            saved[g["group_id"]], _ = EngineGroup.objects.update_or_create(
                owner=owner, source_job=job, group_id=g["group_id"],
                defaults={k: g[k] for k in ("size", "density", "mean_weight", "flagged",
                                            "members", "shared_objects")} | {"cohesion": g.get("cohesion")},
            )
        EngineGroup.objects.filter(owner=owner, source_job=job).exclude(group_id__in=saved).delete()
    return saved


def _save_results(owner, job, result):
    groups = _save_groups(owner, job, result.get("groups") or {})
    for row in result["results"]:
        identity = {
            "owner": owner,
            "platform": row["platform"],
            "external_user_id": row["external_user_id"],
        }
        with transaction.atomic():
            current = (
                EngineAccount.objects.select_for_update(of=("self",))
                .select_related("source_job")
                .filter(**identity)
                .first()
            )
            if (
                current
                and current.source_job
                and current.source_job.created_at > job.created_at
            ):
                continue
            EngineAccount.objects.update_or_create(
                **identity,
                defaults={
                    "username": row.get("username") or "",
                    "score": row["score"],
                    "decision": row["decision"],
                    "model_version": row.get("model_version") or "",
                    "source_job": job,
                    "queue": row.get("queue") or "",
                    "group": groups.get((row.get("group") or {}).get("group_id")),
                    "coordination_score": (row.get("group") or {}).get("coordination_score"),
                    "details": details_from_row(row),
                },
            )


def _sync(owner, job, client, force=False):
    """Bring a job up to date with the engine.

    Results enter CRM once, when the job is first seen completed; status is saved last,
    so an interrupted transfer is repeated on the next visit.
    """
    if not (job.is_active or force):
        return
    state = client.get(f"/v1/jobs/{job.job_id}")
    if job.kind == "analysis" and state["status"] == "completed":
        _save_results(owner, job, client.get(f"/v1/analyses/{job.job_id}"))
    job.status = state["status"]
    job.error = state.get("error") or ""
    job.save(update_fields=["status", "error"])


def _result_rows(result, groups):
    order = {"priority": 0, "standard": 1}
    rows = sorted(result["results"], key=lambda r: (order.get(r.get("queue"), 2), -(r["score"] or 0)))
    out = []
    for row in rows[:SHOWN_RESULTS]:
        group = row.get("group") or {}
        out.append({
            "card": account_card(
                details_from_row(row), platform=row["platform"],
                external_user_id=row["external_user_id"], username=row.get("username") or "",
                decision=row["decision"], score=row["score"], queue=row.get("queue") or ""),
            "group": groups.get(group.get("group_id")) if group.get("flagged") else None,
        })
    return out


@login_required
def detail(request, job_id):
    job = get_object_or_404(EngineJob.objects.select_related("source"), job_id=job_id, owner=request.user)
    client = EngineClient(request.user)
    context = {"job": job, "submission_key": str(uuid.uuid4()), "unavailable": False,
               "analyses": job.analyses.filter(owner=request.user)}
    try:
        _sync(request.user, job, client)
        if job.kind == "analysis" and job.status == "completed":
            result = client.get(f"/v1/analyses/{job.job_id}")
            groups = {g.group_id: g for g in EngineGroup.objects.filter(owner=request.user, source_job=job)}
            context.update(
                rows=_result_rows(result, groups), total=len(result["results"]),
                priority=sum(r.get("queue", "") == "priority" for r in result["results"]),
                flagged_groups=sum(g.flagged for g in groups.values()))
    except EngineClientError as exc:
        context["unavailable"] = str(exc)
    return render(request, "core/engine_job.html", context)


@login_required
def status(request, job_id):
    """Polled by the page while a job runs; reads only."""
    job = get_object_or_404(EngineJob, job_id=job_id, owner=request.user)
    if not job.is_active:
        return JsonResponse({"status": job.status})
    try:
        return JsonResponse({"status": EngineClient(request.user).get(f"/v1/jobs/{job.job_id}")["status"]})
    except EngineClientError:
        return JsonResponse({"status": "unavailable"}, status=503)


@login_required
@require_POST
def refresh(request, job_id):
    """Force a repeat transfer, then show the ordinary page (no result behind a POST)."""
    job = get_object_or_404(EngineJob, job_id=job_id, owner=request.user)
    try:
        _sync(request.user, job, EngineClient(request.user), force=True)
    except EngineClientError as exc:
        messages.error(request, str(exc))
    return redirect("engine_job", job_id=job_id)


@login_required
@require_POST
def analyze(request, job_id):
    source = get_object_or_404(EngineJob, job_id=job_id, owner=request.user)
    try:
        result = EngineClient(request.user).submit(
            "/v1/analyses",
            {"source_job_id": source.job_id},
            request.POST.get("idempotency_key"),
        )
        job = _remember(request.user, result, "analysis", source.title, source)
        if result.get("dispatch_pending"):
            messages.warning(
                request,
                "Задание сохранено. Очередь временно недоступна; администратор сможет восстановить запуск.",
            )
        return redirect("engine_job", job_id=job.job_id)
    except EngineClientError as exc:
        messages.error(request, str(exc))
    return redirect("engine_job", job_id=job_id)


@login_required
def accounts(request):
    queue_first = Case(When(queue="priority", then=0), default=1)
    return render(
        request,
        "core/engine_accounts.html",
        {"accounts": EngineAccount.objects.filter(owner=request.user)
         .select_related("group", "source_job").order_by(queue_first, F("score").desc(nulls_last=True), "pk")[:500]},
    )


@login_required
def groups(request):
    return render(
        request,
        "core/engine_groups.html",
        {"groups": EngineGroup.objects.filter(owner=request.user).select_related("source_job")[:300]},
    )


@login_required
def group_detail(request, pk):
    group = get_object_or_404(EngineGroup.objects.select_related("source_job"), pk=pk, owner=request.user)
    return render(
        request,
        "core/engine_group.html",
        {"group": group, "accounts": group.accounts.order_by(F("score").desc(nulls_last=True), "pk")},
    )


@login_required
@require_POST
def label(request, pk):
    account = get_object_or_404(EngineAccount, pk=pk, owner=request.user)
    value = request.POST.get("label", "unknown")
    if value not in ("bot", "human", "unknown"):
        messages.error(request, "Некорректная метка")
    else:
        account.label = value
        account.notes = request.POST.get("notes", "")[:10000]
        account.save(update_fields=["label", "notes", "updated_at"])
        messages.success(request, "Разметка сохранена")
    return redirect(f"{reverse('engine_accounts')}#account-{account.pk}")
