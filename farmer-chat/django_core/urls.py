"""django_core URL Configuration"""

from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView
from django.http import HttpResponse, JsonResponse

def favicon_view(request):
    return HttpResponse(status=204)

def api_ping(request):
    return JsonResponse({"status": "ok"})

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),

    # Render index.html at root
    path("", TemplateView.as_view(template_name="index.html"), name="index"),

    # Helpers
    path("api/ping/", api_ping),
    path("favicon.ico", favicon_view),
]
