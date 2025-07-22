import ccxt
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import coint
import warnings
import time
import telegram  # For DMs
warnings.filterwarnings('ignore')

# ---------- Configuration ----------
# Your Wallex API Keys (keep secret!)
WALLEX_API_KEY = '15163F5FpOxwB5h0sz6FUcxjDtyjKxEEgCLf1JJRjxtPN'


# Telegram Bot
TELEGRAM_TOKEN = '7993659973:AAER9YPr8H9yEE5TycFCyHTyBKIfLl3JAUw'  # From @BotFather
TELEGRAM_CHAT_ID = '748595017'  # Your user ID

# Trading Params (from your script, adjust based on grid search results)
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Example pair; extend as needed
TIMEFRAME = '4h'
LOOKBACK = 300
FEE = 0.001  # Round-trip fee
COINTEG_THRESHOLD = 0.05
LEVERAGE = 10  # Futures leverage (e.g., 10x)
TRADE_AMOUNT_USD = 100  # Amount per trade in USD
Z_ENTRY = 2.0  # Best from grid search
Z_EXIT = 0.5
WINDOW = 50
STOPLOSS_PCT = 0.04
STOPLOSS_Z = 3.0

# Initialize Exchange
exchange = ccxt.wallex({
    'apiKey': WALLEX_API_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}  # For futures
})
exchange.load_markets()

# Telegram Bot Setup
bot = telegram.Bot(token=TELEGRAM_TOKEN)

def send_telegram_message(message):
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# Fetch Live Data (modified for real-time)
def fetch_live_data(sym):
    try:
        ohlc = exchange.fetch_ohlcv(sym + ':USDT', timeframe=TIMEFRAME, limit=LOOKBACK)  # Futures symbol
        df = pd.DataFrame(ohlc, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {sym}: {e}")
        return pd.DataFrame()

# Find Best Pair (run once at start)
def find_best_pair():
    all_data = {sym: fetch_live_data(sym) for sym in SYMBOLS}
    pairs_stats = []
    for s1, s2 in itertools.combinations(SYMBOLS, 2):
        df1, df2 = all_data[s1], all_data[s2]
        dfm = pd.merge(df1[['datetime', 'close']], df2[['datetime', 'close']], on='datetime')
        if len(dfm) < 100: continue
        score, pval, _ = coint(dfm['close_x'], dfm['close_y'])
        pairs_stats.append({'pair': f"{s1} ~ {s2}", 'pval': pval})
    filtered_pairs = [p for p in pairs_stats if p['pval'] < COINTEG_THRESHOLD]
    return filtered_pairs[0]['pair'] if filtered_pairs else None  # Use first best pair

# Main Trading Loop
def trader_ai_loop():
    best_pair = find_best_pair()
    if not best_pair:
        send_telegram_message("No cointegrated pair found. Stopping.")
        return
    
    s1, s2 = best_pair.split(' ~ ')
    send_telegram_message(f"Trader AI started. Best pair: {best_pair}")
    
    in_position = 0  # 0: no position, 1: long spread, -1: short spread
    entry_price1, entry_price2 = 0, 0
    entry_spread, entry_z = 0, 0
    
    while True:  # Continuous loop
        try:
            df1 = fetch_live_data(s1)
            df2 = fetch_live_data(s2)
            dfm = pd.merge(df1[['datetime', 'close']], df2[['datetime', 'close']], on='datetime')[-LOOKBACK:]
            
            price1 = dfm['close_x'].to_numpy()[-1]
            price2 = dfm['close_y'].to_numpy()[-1]
            spread = pd.Series(dfm['close_x'] - dfm['close_y'])
            zscores = (spread - spread.rolling(WINDOW).mean()) / (spread.rolling(WINDOW).std() + 1e-8)
            current_z = zscores[-1]
            
            if in_position == 0:  # Check for entry
                if current_z > Z_ENTRY:  # Short spread
                    in_position = -1
                    entry_price1, entry_price2 = price1, price2
                    entry_spread = spread[-1]
                    entry_z = current_z
                    # Execute trade (example: short s1, long s2 with leverage)
                    amount1 = (TRADE_AMOUNT_USD / price1) * LEVERAGE  # Size calculation
                    amount2 = (TRADE_AMOUNT_USD / price2) * LEVERAGE
                    exchange.create_market_sell_order(s1 + ':USDT', amount1)  # Short s1
                    exchange.create_market_buy_order(s2 + ':USDT', amount2)   # Long s2
                    send_telegram_message(f"Started trade on {best_pair} with {TRADE_AMOUNT_USD} USD. Direction: Short Spread. Leverage: {LEVERAGE}x.")
                
                elif current_z < -Z_ENTRY:  # Long spread
                    in_position = 1
                    entry_price1, entry_price2 = price1, price2
                    entry_spread = spread[-1]
                    entry_z = current_z
                    amount1 = (TRADE_AMOUNT_USD / price1) * LEVERAGE
                    amount2 = (TRADE_AMOUNT_USD / price2) * LEVERAGE
                    exchange.create_market_buy_order(s1 + ':USDT', amount1)   # Long s1
                    exchange.create_market_sell_order(s2 + ':USDT', amount2)  # Short s2
                    send_telegram_message(f"Started trade on {best_pair} with {TRADE_AMOUNT_USD} USD. Direction: Long Spread. Leverage: {LEVERAGE}x.")
            
            elif in_position != 0:  # Check for exit/stop-loss
                current_spread = price1 - price2
                raw_pnl = (current_spread - entry_spread) * in_position
                raw_pnl_pc = raw_pnl / abs(entry_spread + 1e-8)
                exit_reason = None
                
                if raw_pnl_pc < -STOPLOSS_PCT:
                    exit_reason = 'STOP_PCT'
                elif abs(current_z - entry_z) > STOPLOSS_Z:
                    exit_reason = 'STOP_Z'
                elif abs(current_z) < Z_EXIT:
                    exit_reason = 'NORMAL'
                
                if exit_reason:
                    # Close positions
                    if in_position == 1:
                        exchange.create_market_sell_order(s1 + ':USDT', amount1)  # Close long s1
                        exchange.create_market_buy_order(s2 + ':USDT', amount2)   # Close short s2
                    else:
                        exchange.create_market_buy_order(s1 + ':USDT', amount1)   # Close short s1
                        exchange.create_market_sell_order(s2 + ':USDT', amount2)  # Close long s2
                    
                    pnl_after_fee = raw_pnl - (entry_price1 + entry_price2) * FEE * LEVERAGE
                    pnl_pct = (pnl_after_fee / TRADE_AMOUNT_USD) * 100
                    result = 'Gain' if pnl_after_fee > 0 else 'Loss'
                    send_telegram_message(f"Ended trade on {best_pair}. Result: {result} {abs(pnl_after_fee):.2f} USD ({pnl_pct:.2f}%). Reason: {exit_reason}.")
                    in_position = 0
            
            time.sleep(60 * 60 * 4)  # Sleep 4 hours (match TIMEFRAME), adjust for faster checks
        
        except Exception as e:
            print(f"Error in loop: {e}")
            send_telegram_message(f"Error in Trader AI: {e}")
            time.sleep(60)  # Retry after 1 min

# Run the Trader AI
if __name__ == "__main__":
    trader_ai_loop()
