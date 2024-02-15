from django.contrib import admin 
from django.urls import path 
from chartjs import views 
  
urlpatterns = [ 
    path('admin/', admin.site.urls), 
    path('', views.HomeView.as_view()), 
    path('get_flask_data/', views.get_flask_data, name='get_flask_data'),
    path('api', views.ChartData.as_view()), 
] 