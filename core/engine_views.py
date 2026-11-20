import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import F
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .engine_client import EngineClient, EngineClientError
from .models import EngineAccount, EngineJob


def _remember(owner, result, kind):
    job, _ = EngineJob.objects.get_or_create(
        job_id=result["job_id"],
        owner=owner,
        defaults={"kind": kind, "status": result["status"]},
    )
    return job


@login_required
def jobs(request):
    return render(
        request,
        "core/engine_jobs.html",
        {
            "jobs": EngineJob.objects.filter(owner=request.user)[:100],
            "submission_key": str(uuid.uuid4()),
        },
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
        job = _remember(request.user, result, "import")
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
        job = _remember(request.user, result, "threads")
        if result.get("dispatch_pending"):
            messages.warning(
                request,
                "Задание сохранено. Очередь временно недоступна; администратор сможет восстановить запуск.",
            )
        return redirect("engine_job", job_id=job.job_id)
    except EngineClientError as exc:
        messages.error(request, str(exc))
    return redirect("engine_jobs")


@login_required
def detail(request, job_id):
    job = get_object_or_404(EngineJob, job_id=job_id, owner=request.user)
    return render(
        request,
        "core/engine_job.html",
        {"job": job, "submission_key": str(uuid.uuid4())},
    )


@login_required
@require_POST
def refresh(request, job_id):
    job = get_object_or_404(EngineJob, job_id=job_id, owner=request.user)
    client = EngineClient(request.user)
    try:
        status = client.get(f"/v1/jobs/{job.job_id}")
        job.status = status["status"]
        job.error = status.get("error") or ""
        job.save(update_fields=["status", "error"])
        if job.kind == "analysis" and job.status == "completed":
            result = client.get(f"/v1/analyses/{job.job_id}")
            for row in result["results"]:
                identity = {
                    "owner": request.user,
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
                        },
                    )
            return render(
                request,
                "core/engine_job.html",
                {"job": job, "result": result, "submission_key": str(uuid.uuid4())},
            )
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
        job = _remember(request.user, result, "analysis")
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
    return render(
        request,
        "core/engine_accounts.html",
        {"accounts": EngineAccount.objects.filter(owner=request.user).order_by(F("score").desc(nulls_last=True), "pk")[:500]},
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
    return redirect("engine_accounts")
