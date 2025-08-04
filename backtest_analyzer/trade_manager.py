from binance.error import ClientError
from .models import CoinPairsList, Trade
import pandas as pd
from time import sleep
import datetime
MAX_CONSECUTIVE_LOSSES = 3
VOLUME = 2.6 # volume for one order (if its 10 and leverage is 10, then you put your 1 usdt to one position)
LEVERAGE = 2      # total usdt is 5*2=10 usdt
ORDER_TYPE = 'ISOLATED'  # type is 'ISOLATED' or 'CROSS'

def get_balance_usdt(client):
    print("----fetching Balance")
    try:
        response = client.balance(recvWindow=10000)
        for elem in response:
            if elem['asset'] == 'USDT':
                print("balance - ",float(elem['balance']))
                return float(elem['balance'])

    except ClientError as error:
        print(
            "----fetching Balance Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Set leverage for the needed symbol. You need this bcz different symbols can have different leverage
def set_leverage(client,symbol, level):
    print("----setting Leverage")
    try:
        response = client.change_leverage(
            symbol=symbol, leverage=level, recvWindow=6000
        )
        print(response)
    except ClientError as error:
        print(
            "----setting Leverage Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# The same for the margin type
def set_mode(client, symbol, order_type):
    print("----Setting Mode ")
    try:
        response = client.change_margin_type(
            symbol=symbol, marginType=order_type, recvWindow=6000
        )
        print(response)
    except ClientError as error:
        print(
            "----Setting Mode Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Price precision. BTC has 1, XRP has 4
def get_price_precision(client, symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['pricePrecision']

# Amount precision. BTC has 3, XRP has 1
def get_qty_precision(client, symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['quantityPrecision']

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


def analyze_trades(trades):
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
    # print("all trades")
    # print(f"{trades_df}")
    # Filter real trades
    real_trades_df = trades_df[~trades_df['is_virtual']].copy()
    # Calculate consecutive wins and losses for real trades
    current_wins = 0
    current_losses = 0
    winloss_data=[]
    for _, trade in real_trades_df.iterrows():
        if trade['result'] == 'win':
            current_wins += 1
            if current_losses>0:
                winloss_data.append({'type': 'losses', 'count': current_losses})
            current_losses = 0
        elif trade['result'] == 'lose':
            current_losses += 1
            if current_wins>0:
                winloss_data.append({'type': 'wins', 'count': current_wins})
            current_wins = 0

    # Don't forget to add the last streak
    if current_wins > 0:
        winloss_data.append({'type': 'wins', 'count': current_wins})
    elif current_losses > 0:
        winloss_data.append({'type': 'losses', 'count': current_losses})

    #print(f"win loss data - {winloss_data}")
    capital_multiplier = get_multiplier(winloss_data)
    last_trade_is_completed = False
    if not real_trades_df.empty:
        last_trade_is_completed = real_trades_df.iloc[-1]['trade_close_time'] is not None
    #print(f"Last trade is {'completed' if last_trade_is_completed else 'not completed'}")
    last_trade = real_trades_df.iloc[-1]
    
    # format  {"side":'sell',"BUY_PRICE":"BUY_PRICE", "SL":"SL","TP":"TP", "SL_Trigger":"SL_Trigger", "TP_Trigger":"TP_Trigger"}
    trade_data = {}
    trade_data["side"] = "buy" if last_trade["side"] == "Buy" else "sell"
    trade_data["BUY_PRICE"] = last_trade['buy_price']
    trade_data["SL"] = last_trade['sl']
    trade_data["SL_Trigger"] = last_trade['sl']
    trade_data["TP"] = last_trade['tp']
    trade_data["TP_Trigger"] = last_trade['tp']
        



    return  last_trade_is_completed, capital_multiplier,  trade_data
    

# Your current positions (returns the symbols list):
def get_pos(client):
    #print("----Getting Positions ")
    #models.BotLogs(description="----Getting Positions ").save()
    try:
        resp = client.get_position_risk()
        pos = []
        for elem in resp:
            if float(elem['positionAmt']) != 0:
                pos.append(elem['symbol'])
        return pos
    except ClientError as error:
        print(
            "----Getting Positions Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


def check_orders(client):
    #print("----Checking Orders ")
    try:
        response = client.get_orders(recvWindow=10000)
        sym = []
        for elem in response:
            sym.append(elem)
        #print("working")
        return sym
    except ClientError as error:
        print(
            "----Checking Orders Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )
       
 # Close open orders for the needed symbol. If one stop order is executed and another one is still there


def close_open_orders(client,symbol):
    print("----Closing Open Orders")
    try:
        response = client.cancel_open_orders(symbol=symbol, recvWindow=10000)
        return response
    except ClientError as error:
        print(
            "----Closing Open Orders Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )       


def remove_pending_orders_repeated(client):
    print("----Removing Pending Orders ")
    #while True:
    try:
        pos = get_pos(client)
        ord = check_orders(client)
        # removing stop orders for closed positions
        for elem in ord:
            if (not elem['symbol'] in pos):  #and (elem['type'] not in ['MARKET','LIMIT']):
                print(elem, "order removed by pending order close function")
                sleep(1)
                close_open_orders(client, elem['symbol'])
        #sleep(60)
    except ClientError as error:
        print(
            "----Removing Pending Orders  Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )
        #sleep(60)
    except:
        #sleep(60)
        pass



# Open new order with the last price, and set TP and SL:
def place_order(client,signal,amount):
    # signal =['coinpair', {"side":'sell',"BUY_PRICE":BUY_PRICE, "SL":SL,"TP":TP}]
    print(f"----Placing Orders for ----- {signal[0]}")
    symbol=signal[0]
    price = float(client.ticker_price(symbol)['price'])
    #print("current price ",price)
    qty_precision = get_qty_precision(client, symbol)
    #print("qty_precision ", qty_precision)
    #price_precision = get_price_precision(client, symbol)
    #print("price precision",price_precision)
    qty = round(amount/price, qty_precision)
    #print("qty", qty)
    if signal[1]['side'] == 'buy':
        try:
            #Limit_price = signal[1]['BUY_PRICE']
            #Limit_price_Trigger = signal[1]['BUY_PRICE_Trigger']
            resp1 = client.new_order(symbol=symbol, side='BUY', type='MARKET', quantity=qty) #price= Limit_price, stopPrice= Limit_price_Trigger, timeInForce='GTC')
            print("Order placed for ",symbol, signal[1]['side'])
            print(resp1)
            sleep(2)
            sl_price = signal[1]['SL']
            sl_price_trigger = signal[1]['SL_Trigger']
            resp2 = client.new_order(symbol=symbol, side='SELL', type='STOP_MARKET', stopPrice=sl_price, closePosition=True) 
            # timeInForce='GTC',stopPrice=sl_price_trigger, price=sl_price) #closePosition=True)
            
            print(f"SL Order Placed for {symbol}")
            print(resp2)
            sleep(2)
            tp_price = signal[1]['TP']
            tp_price_trigger = signal[1]['TP_Trigger']
            resp3 = client.new_order(symbol=symbol, side='SELL', type='TAKE_PROFIT_MARKET',
                                     stopPrice=tp_price_trigger, closePosition=True) 
            #, timeInForce='GTC',closePosition=True,, price=tp_price)
            
            print(f"TP Order Placed for {symbol}")
            print(resp3)
            

        except ClientError as error:
            print(
                "----Placing Orders buy side  Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
            
    if signal[1]['side'] == 'sell':
        try:
            #Limit_price = signal[1]['BUY_PRICE']
            #Limit_price_Trigger = signal[1]['BUY_PRICE_Trigger']
            resp1 = client.new_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty) # Price= Limit_price, stopPrice= Limit_price_Trigger, timeInForce='GTC')
            print(f"Order placed for {symbol} {signal[1]['side']} Side")
            print(resp1)
            sleep(2)
            sl_price = signal[1]['SL']
            #sl_price_trigger = signal[1]['SL_Trigger']
            resp2 = client.new_order(symbol=symbol, side='BUY', type='STOP_MARKET', stopPrice=sl_price,  closePosition=True) 
            #price=sl_price, timeInForce='GTC', stopPrice=sl_price_trigger, price=sl_price)#closePosition=True)
            #, workingType="CONTRACT_PRICE" or MARK_PRICE
            print(f"SL Order Placed for {symbol}")
            print(resp2)
            sleep(2)
            tp_price = signal[1]['TP']
            tp_price_trigger = signal[1]['TP_Trigger']
            resp3 = client.new_order(symbol=symbol, side='BUY', type='TAKE_PROFIT_MARKET', stopPrice=tp_price_trigger,closePosition=True) 
            #price=tp_price,timeInForce='GTC',closePosition=True)
            
            print(f"TP Order Placed for {symbol}")
            print(resp3)
            
        except ClientError as error:
            print(
                "----Placing Orders sell side Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
            


def trade_master(client):
    print("-----Trade master analyzing the pending trades")
    # Fetch all coin pairs from the database
    coin_pairs = CoinPairsList.objects.filter(is_active=True)
    for coin_pair in coin_pairs:
        print(f"checking trades for - {coin_pair.coinpair_name}")
        trades = Trade.objects.filter(coinpair_name=coin_pair.coinpair_name).order_by('trade_start_time')
        if trades.exists():
            #check if trade is already placed or not
            pos = get_pos(client)  
            if coin_pair.coinpair_name not in pos:
                last_trade_is_completed, capital_multiplier, trade_data = analyze_trades(trades)
                print(f"capital multiplier for {coin_pair} - {capital_multiplier} and last trade is completed  - {last_trade_is_completed}")
                if not last_trade_is_completed:
                    ord = check_orders(client)
                    for elem in ord:
                        if not elem['symbol'] in pos:
                            close_open_orders(client, elem['symbol'])
                    print(f"Processing Order for {coin_pair} {trade_data['side']} side with TP - {trade_data['TP']} and SL - {trade_data['SL']}")
                    if get_balance_usdt(client)> 0:
                        set_mode(client, coin_pair, ORDER_TYPE)
                        set_leverage(client, coin_pair, capital_multiplier)  
                        amount = VOLUME * capital_multiplier  
                        place_order(client,[coin_pair,trade_data],amount)
                        print("order placed for {0} and total money invested {1}, leverage {2} ".format(coin_pair,amount,LEVERAGE))
                    else:
                        print("USDT balance is low.... Please add usdt in futures account.")
            else:
                print(f"Trade already exist for - {coin_pair.coinpair_name}")
    remove_pending_orders_repeated(client)
                    

                




