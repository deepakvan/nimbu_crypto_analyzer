from django.contrib import admin
from .models import Trade, CoinPairsList


# Register your models here.
admin.site.register(Trade)
admin.site.register(CoinPairsList)