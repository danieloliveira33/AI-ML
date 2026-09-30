from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("mcp-chat/", views.mcp_chat, name="mcp_chat"),
]