from django.db import models

# Create your models here.
class Trade(models.Model):
    coinpair_name = models.CharField(max_length=50)
    trade_start_time = models.DateTimeField()
    trade_close_time = models.DateTimeField(null=True, blank=True)
    buy_price = models.DecimalField(max_digits=20, decimal_places=8)
    tp = models.DecimalField(max_digits=20, decimal_places=8)
    sl = models.DecimalField(max_digits=20, decimal_places=8)
    side = models.CharField(max_length=10)
    result = models.CharField(max_length=20,null=True, blank=True)  # 'won', 'lost', or None
    gain_percentage = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.coinpair_name} ({self.trade_start_time})"


class CoinPairsList(models.Model):
    coinpair_name = models.CharField(max_length=50, unique=True)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return self.coinpair_name