from . import views
from django.urls import path

urlpatterns = [
    #path('', views.home),
    path('', views.analytics_page, name='analytics'),
    path('api/trade-analytics/<str:coin_pair>/', views.TradeAnalyticsView.as_view(), name='trade-analytics'),
    path('api/trade-analytics/', views.TradeAnalyticsView.as_view(), name='coin-pairs-list'),
]


import threading
from .views import bot
threading.Thread(target=bot).start()



