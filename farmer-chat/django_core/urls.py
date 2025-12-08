from django.contrib import admin
from django.conf import settings
from django.conf. urls.static import static
from django. urls import path, include
from django.views.generic import TemplateView
from django.http import HttpResponse, JsonResponse

def favicon_view(request):
    return HttpResponse(status=204)

def api_ping(request):
    return JsonResponse({"status": "ok"})

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    path("api/profile/", include("user_profile.urls")),    
    path("", TemplateView.as_view(template_name="index.html"), name="index"),
    
    path("api/ping/", api_ping),
    path("favicon.ico", favicon_view),
    path('api/skin-analysis/', include('skin_analysis.urls')),
    path('api/lab-report/', include('lab_report.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings. MEDIA_URL, document_root=settings.MEDIA_ROOT)
