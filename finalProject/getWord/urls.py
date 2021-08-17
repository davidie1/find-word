from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detectWord/', views.detectWord, name='detectWord'),
    path('show/', views.show, name='show'),
]