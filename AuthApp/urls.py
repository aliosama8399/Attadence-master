# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('login', views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
    path('', views.index, name='index'),
    path('takeAttadence',views.takeAttendence, name='takeAttendence'),
    path('load-training-images/', views.load_training_images, name='load_training_images'),
    

]
