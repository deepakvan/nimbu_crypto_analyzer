from django.shortcuts import render
from django.http import HttpResponse
import os
from django.http import HttpResponse
from django.shortcuts import render
from . import models
import datetime
import time
from time import sleep
from binance.um_futures import UMFutures
import pandas as pd
import threading
import random
from .models import CoinPairsList, Trade
from . import helper_functions as hf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from . import trade_manager

# Strategy parameters
MAX_CONSECUTIVE_LOSSES = 3
BROKERAGE_RATE = 0.001


# Binance API credentials (ensure these are stored securely, e.g., in settings.py)
HISTORY_API_KEY = settings.HISTORY_API_KEY
HISTORY_API_SECRET = settings.HISTORY_API_SECRET
#history_client = UMFutures(key=HISTORY_API_KEY, secret=HISTORY_API_SECRET)
history_client = UMFutures()

#API_KEY = settings.API_KEY
#API_SECRET = settings.API_SECRET
#client = UMFutures(key=API_KEY, secret=API_SECRET)





# Create your views here.
def home(request):
    """
    Render the home page of the backtester application.
    """
    return HttpResponse("Welcome to the Backtester Home Page")

def get_multiplier(winloss_data):
    #[{'type': 'losses', 'count': 2}, {'type': 'wins', 'count': 3}, {'type': 'losses', 'count': 1}, {'type': 'wins', 'count': 1}]
    MAX_LOSS_COUNTER = 3
    pending_losses=0
    for list_element in winloss_data:
        if list_element['type']=='losses':
            pending_losses+=list_element['count']
        if list_element['type']=='wins':
            if pending_losses<=MAX_LOSS_COUNTER:
                pending_losses=0
                continue
            for i in range(list_element['count']):
                if pending_losses<=MAX_LOSS_COUNTER:
                    pending_losses=0
                    break
                pending_losses-=1

    if pending_losses>=(MAX_LOSS_COUNTER-1):
        return 2**MAX_LOSS_COUNTER
    else:
        return 2**(pending_losses+1)

def calculate_trade_outcomes(trades):
    """
    Calculate trade outcomes and determine virtual trades at runtime.
    
    Virtual trade logic:
    - After 3 consecutive real trade losses, subsequent trades are virtual.
    - Virtual trades continue until a virtual trade wins.
    - After a virtual win, the next trade is real, and the real loss counter resets.
    - If a real trade loses, increment the real loss counter; if it reaches 4 or more, virtual trades resume until one virtual trade wins.

    Args:
        trades: Queryset of Trade objects for a coin pair
    
    Returns:
        Dictionary with trading statistics
    """
    # Convert trades to DataFrame
    trades_list = [
        {
            'trade_start_time': trade.trade_start_time.isoformat(),
            'trade_close_time': trade.trade_close_time.isoformat() if trade.trade_close_time else None,
            'buy_price': float(trade.buy_price),
            'tp': float(trade.tp),
            'sl': float(trade.sl),
            'side': trade.side,
            'result': trade.result,
            'gain_percentage': trade.gain_percentage
        }
        for trade in trades
    ]
    trades_df = pd.DataFrame(trades_list)
    
    #print(f"Trades data for analysis: {trades_list}")  # Debug: Log trades data
    
    if trades_df.empty:
        return {
            'total_trades': 0,
            'real_trades': 0,
            'virtual_trades': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'real_win_trades': 0,
            'real_lose_trades': 0,
            'buy_total': 0,
            'buy_win_trades': 0,
            'buy_lose_trades': 0,
            'sell_total': 0,
            'sell_win_trades': 0,
            'sell_lose_trades': 0,
            'buy_win_pct': 0,
            'sell_win_pct': 0,
            'overall_win_pct': 0,
            'net_profit_pct': 0,
            'gross_profit_pct': 0,
            'brokerage_pct': 0,
            'trades': []
        }
    
    # Determine virtual trades
    is_virtual_list = []
    consecutive_real_losses = 0
    is_virtual = False
    
    for _, trade in trades_df.iterrows():
        if is_virtual:
            # Currently in virtual mode
            is_virtual_list.append(True)
            if trade['result'] == 'win':
                # Virtual trade won, next trade is real
                is_virtual = False
        else:
            # Currently in real mode
            is_virtual_list.append(False)
            if trade['result'] == 'lose':
                consecutive_real_losses += 1
                if consecutive_real_losses >= MAX_CONSECUTIVE_LOSSES:
                    # After 3 consecutive real losses, switch to virtual
                    is_virtual = True
            else:
                # Real trade won, reset loss counter
                consecutive_real_losses = 0
    
    trades_df['is_virtual'] = is_virtual_list
    # Filter real and virtual trades
    real_trades_df = trades_df[~trades_df['is_virtual']].copy()
    virtual_trades_df = trades_df[trades_df['is_virtual']].copy()
    
    # Basic counts
    total_trades = len(trades_df)
    real_trades_count = len(real_trades_df)
    virtual_trades_count = len(virtual_trades_df)
    
    # Calculate consecutive wins and losses for real trades
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    winloss_data=[]

    for _, trade in real_trades_df.iterrows():
        if trade['result'] == 'win':
            current_wins += 1
            if current_losses>0:
                winloss_data.append({'type': 'losses', 'count': current_losses})
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        elif trade['result'] == 'lose':
            current_losses += 1
            if current_wins>0:
                winloss_data.append({'type': 'wins', 'count': current_wins})
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    # Don't forget to add the last streak
    if current_wins > 0:
        winloss_data.append({'type': 'wins', 'count': current_wins})
    elif current_losses > 0:
        winloss_data.append({'type': 'losses', 'count': current_losses})

    capital_multiplier = get_multiplier(winloss_data)

    # Real trade statistics
    real_win_trades = len(real_trades_df[real_trades_df['result'] == 'win'])
    real_lose_trades = len(real_trades_df[real_trades_df['result'] == 'lose'])
    
    # Buy/Sell breakdown for real trades
    real_buy_trades = real_trades_df[real_trades_df['side'] == 'Buy']
    real_sell_trades = real_trades_df[real_trades_df['side'] == 'Sell']
    
    buy_win_trades = len(real_buy_trades[real_buy_trades['result'] == 'win'])
    buy_lose_trades = len(real_buy_trades[real_buy_trades['result'] == 'lose'])
    buy_total = len(real_buy_trades)
    
    sell_win_trades = len(real_sell_trades[real_sell_trades['result'] == 'win'])
    sell_lose_trades = len(real_sell_trades[real_sell_trades['result'] == 'lose'])
    sell_total = len(real_sell_trades)
    
    # Win percentages
    buy_win_pct = (buy_win_trades / buy_total * 100) if buy_total > 0 else 0
    sell_win_pct = (sell_win_trades / sell_total * 100) if sell_total > 0 else 0
    overall_win_pct = (real_win_trades / real_trades_count * 100) if real_trades_count > 0 else 0
    
    # Profit calculations
    gross_profit_pct = real_trades_df['gain_percentage'].sum() if not real_trades_df.empty else 0
    brokerage_pct = real_trades_count * BROKERAGE_RATE * 100 * 2  # Entry + exit
    net_profit_pct = gross_profit_pct - brokerage_pct
    
    # Prepare trades data for frontend
    trades_data = trades_df.to_dict('records')
    
    return {
        'total_trades': total_trades,
        'real_trades': real_trades_count,
        'virtual_trades': virtual_trades_count,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'real_win_trades': real_win_trades,
        'real_lose_trades': real_lose_trades,
        'buy_total': buy_total,
        'buy_win_trades': buy_win_trades,
        'buy_lose_trades': buy_lose_trades,
        'sell_total': sell_total,
        'sell_win_trades': sell_win_trades,
        'sell_lose_trades': sell_lose_trades,
        'buy_win_pct': round(buy_win_pct, 1),
        'sell_win_pct': round(sell_win_pct, 1),
        'overall_win_pct': round(overall_win_pct, 1),
        'net_profit_pct': round(net_profit_pct, 1),
        'gross_profit_pct': round(gross_profit_pct, 1),
        'brokerage_pct': round(brokerage_pct, 1),
        'trades': trades_data,
        'capital_multiplier':capital_multiplier
    }

class TradeAnalyticsView(APIView):
    """
    API view to fetch analytics for a specific coin pair.
    """
    def get(self, request, coin_pair=None):
        if coin_pair:
            trades = Trade.objects.filter(coinpair_name=coin_pair).order_by('trade_start_time')
            if not trades.exists():
                return Response({'error': f'No trades found for {coin_pair}'}, status=status.HTTP_404_NOT_FOUND)
            analytics = calculate_trade_outcomes(trades)
            return Response(analytics)
        else:
            coin_pairs = CoinPairsList.objects.all().values_list('coinpair_name', flat=True)
            return Response({'coin_pairs': list(coin_pairs)})

def analytics_page(request):
    """
    Render the analytics page with coin pairs list.
    """
    coin_pairs = CoinPairsList.objects.all()
    return render(request, 'analytics.html', {'coin_pairs': coin_pairs})


def bot():
    print("Starting the backtester bot............")
    coin_pairs = CoinPairsList.objects.all()
    while True:
        try:
            seconds = datetime.datetime.now().second
            if seconds>5 and seconds<7:
                print(f"Starting backtest for {len(coin_pairs)} coin pairs...")
                for coin_pair in coin_pairs:
                    hf.process_coin_pair(coin_pair.coinpair_name, history_client)

                print("Backtest completed for all coin pairs. sleeping for 1 hour...")
                sleep(60*60)  # Sleep for 1 hour before the next iteration previously it was 30 seconds
        except:
            print("Error in bot function Code")
