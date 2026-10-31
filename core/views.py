from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from analysis.features import BEHAVIOR_FEATURES, NETWORK_FEATURES
from analysis.graph import DEFAULT_WEIGHTS, GraphParams
from analysis.pipeline import run_analysis
from analysis.synthetic import generate

from .importer import import_records, read_csv
from .models import Account, AnalysisRun, Community, Edge, Post


def dashboard(request):
    run = AnalysisRun.objects.first()
    ctx = {
        "run": run,
        "n_accounts": Account.objects.count(),
        "n_posts": Post.objects.count(),
        "n_high": Account.objects.filter(risk__gte=0.7).count(),
        "n_medium": Account.objects.filter(risk__gte=0.4, risk__lt=0.7).count(),
        "communities": run.communities.all()[:6] if run else [],
        "top_accounts": Account.objects.exclude(risk=None)[:8],
        "weights": DEFAULT_WEIGHTS,
    }
    return render(request, "core/dashboard.html", ctx)


def accounts(request):
    qs = Account.objects.select_related("community").annotate(n_posts=Count("posts")).order_by("-risk", "handle")
    q = request.GET.get("q", "").strip()
    level = request.GET.get("level", "")
    if q:
        qs = qs.filter(Q(handle__icontains=q) | Q(display_name__icontains=q))
    if level == "high":
        qs = qs.filter(risk__gte=0.7)
    elif level == "medium":
        qs = qs.filter(risk__gte=0.4, risk__lt=0.7)
    elif level == "low":
        qs = qs.filter(risk__lt=0.4)
    page = Paginator(qs, 40).get_page(request.GET.get("page"))
    return render(request, "core/accounts.html", {"page": page, "q": q, "level": level})


def account_detail(request, handle):
    acc = get_object_or_404(Account.objects.select_related("community"), handle=handle)
    f = acc.features or {}
    edges = Edge.objects.filter(Q(a=acc) | Q(b=acc)).select_related("a", "b").order_by("-weight")[:15]
    neighbours = [{"other": e.b if e.a_id == acc.id else e.a, "weight": e.weight, "c": e.components} for e in edges]
    return render(request, "core/account_detail.html", {
        "acc": acc,
        "behavior": [(k, f.get(k)) for k in BEHAVIOR_FEATURES],
        "network": [(k, f.get(k)) for k in NETWORK_FEATURES],
        "posts": acc.posts.all()[:30],
        "neighbours": neighbours,
    })


def communities(request):
    run = AnalysisRun.objects.first()
    items = run.communities.all() if run else []
    return render(request, "core/communities.html", {"run": run, "items": items})


def community_detail(request, pk):
    com = get_object_or_404(Community, pk=pk)
    members = com.members.annotate(n_posts=Count("posts")).order_by("-risk")
    return render(request, "core/community_detail.html", {"com": com, "members": members,
                                                           "stats": members.aggregate(Avg("risk"), Avg("bot_proba"))})


def graph(request):
    return render(request, "core/graph.html", {"run": AnalysisRun.objects.first()})


def data(request):
    return render(request, "core/data.html", {"runs": AnalysisRun.objects.all()[:10]})


@require_POST
def upload(request):
    try:
        acc = read_csv(request.FILES["accounts"]) if "accounts" in request.FILES else []
        posts = read_csv(request.FILES["posts"]) if "posts" in request.FILES else []
        na, np_ = import_records(acc, posts, platform=request.POST.get("platform", "threads"))
        messages.success(request, f"Импортировано: {na} аккаунтов, {np_} постов")
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"Ошибка импорта: {e}")
    return redirect("data")


@require_POST
def synthetic(request):
    if request.POST.get("reset"):
        Account.objects.all().delete()
    a, p = generate(seed=int(request.POST.get("seed") or 42))
    na, np_ = import_records(a, p, platform="synthetic")
    messages.success(request, f"Синтетика: {na} аккаунтов, {np_} постов")
    return redirect("data")


@require_POST
def analyze(request):
    try:
        params = GraphParams(threshold=float(request.POST.get("threshold") or 0.30),
                             window_s=int(request.POST.get("window") or 600))
        run = run_analysis(params, retrain=bool(request.POST.get("retrain")))
        messages.success(request, f"Анализ завершён: {run.n_communities} сообществ, {run.n_edges} рёбер")
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"Ошибка анализа: {e}")
    return redirect("dashboard")


@require_POST
def reset(request):
    Account.objects.all().delete()
    AnalysisRun.objects.all().delete()
    messages.info(request, "Данные удалены")
    return redirect("data")


# ---------- JSON API ----------

def api_accounts(request):
    qs = Account.objects.exclude(risk=None).values("handle", "label", "bot_proba", "anomaly_score", "risk",
                                                   "community_id", "dbscan_cluster")[: int(request.GET.get("limit", 500))]
    return JsonResponse({"results": list(qs)})


def api_account(request, handle):
    a = get_object_or_404(Account, handle=handle)
    return JsonResponse({"handle": a.handle, "label": a.label, "bot_proba": a.bot_proba, "anomaly_score": a.anomaly_score,
                         "risk": a.risk, "community": a.community_id, "features": a.features, "explanation": a.explanation})


def api_communities(request):
    run = AnalysisRun.objects.first()
    qs = run.communities.values() if run else []
    return JsonResponse({"run": run.pk if run else None, "results": list(qs)})


def api_graph(request):
    run = AnalysisRun.objects.first()
    if not run:
        return JsonResponse({"nodes": [], "edges": []})
    min_w = float(request.GET.get("min_weight", 0))
    edges = list(run.edges.filter(weight__gte=min_w).values("a_id", "b_id", "weight", "components"))
    ids = {e["a_id"] for e in edges} | {e["b_id"] for e in edges}
    nodes = list(Account.objects.filter(id__in=ids).values("id", "handle", "risk", "community_id", "label"))
    return JsonResponse({"nodes": nodes, "edges": edges})
