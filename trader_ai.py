import ccxt
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import coint
import warnings
import time
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import threading
warnings.filterwarnings('ignore')

# ---------- Configuration ----------
# Your Wallex API Keys (keep secret!) - Not used in paper trading mode
WALLEX_API_KEY = '15163F5FpOxwB5h0sz6FUcxjDtyjKxEEgCLf1JJRjxtPN'

# Telegram Bot
TELEGRAM_TOKEN = '7993659973:AAER9YPr8H9yEE5TycFCyHTyBKIfLl3JAUw'  # From @BotFather
TELEGRAM_CHAT_ID = '748595017'  # Your user ID (must match for security)

# Trading Params (from your script, adjust based on grid search results)
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Example pair; extend as needed
TIMEFRAME = '4h'
LOOKBACK = 300
FEE = 0.001  # Round-trip fee (simulated in paper mode)
COINTEG_THRESHOLD = 0.05
LEVERAGE = 10  # Futures leverage (e.g., 10x) - simulated
TRADE_AMOUNT_USD = 100  # Amount per trade in USD - virtual in paper mode

# *** NEW: Paper Trading Mode (set to True for unreal/test money) ***
PAPER_TRADING = True  # Toggle to False for real trading

Z_ENTRY = 2.0  # Best from grid search
Z_EXIT = 0.5
WINDOW = 50
STOPLOSS_PCT = 0.04
STOPLOSS_Z = 3.0

# Global Flags for Start/Stop and Position Tracking
is_trading_active = False
trading_thread = None
in_position = 0  # 0: no position, 1: long spread, -1: short spread
current_pair = None
s1, s2 = None, None
entry_price1, entry_price2 = 0, 0
entry_spread, entry_z = 0, 0
simulated_amount1, simulated_amount2 = 0, 0  # For paper mode

# Initialize Exchange (used for data fetching even in paper mode)
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

# Fetch Live Data
def fetch_live_data(sym):
    try:
        ohlc = exchange.fetch_ohlcv(sym + ':USDT', timeframe=TIMEFRAME, limit=LOOKBACK)  # Futures symbol
        df = pd.DataFrame(ohlc, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {sym}: {e}")
        return pd.DataFrame()

# Find Best Pair
def find_best_pair():
    all_data = {sym: fetch_live_data(sym) for sym in SYMBOLS}
    pairs_stats = []
    for sym1, sym2 in itertools.combinations(SYMBOLS, 2):
        df1, df2 = all_data[sym1], all_data[sym2]
        dfm = pd.merge(df1[['datetime', 'close']], df2[['datetime', 'close']], on='datetime')
        if len(dfm) < 100: continue
        score, pval, _ = coint(dfm['close_x'], dfm['close_y'])
        pairs_stats.append({'pair': f"{sym1} ~ {sym2}", 'pval': pval})
    filtered_pairs = [p for p in pairs_stats if p['pval'] < COINTEG_THRESHOLD]
    return filtered_pairs[0]['pair'] if filtered_pairs else None

# Simulate Order (for paper trading)
def simulate_order(action, sym, amount, price):
    print(f"SIMULATED {action} on {sym} for {amount} at price {price}")
    # You could add more simulation logic here (e.g., slippage), but keeping it simple

# Close Position Safely
def close_position(exit_reason='MANUAL_STOP'):
    global in_position, simulated_amount1, simulated_amount2, s1, s2
    if in_position != 0 and s1 and s2:
        price1 = exchange.fetch_ticker(s1 + ':USDT')['close']
        price2 = exchange.fetch_ticker(s2 + ':USDT')['close']
        current_spread = price1 - price2
        raw_pnl = (current_spread - entry_spread) * in_position
        pnl_after_fee = raw_pnl - (entry_price1 + entry_price2) * FEE * LEVERAGE
        pnl_pct = (pnl_after_fee / TRADE_AMOUNT_USD) * 100
        result = 'Gain' if pnl_after_fee > 0 else 'Loss'
        
        if PAPER_TRADING:
            simulate_order('CLOSE', s1, simulated_amount1, price1)
            simulate_order('CLOSE', s2, simulated_amount2, price2)
        else:
            if in_position == 1:
                exchange.create_market_sell_order(s1 + ':USDT', simulated_amount1)  # Close long s1
                exchange.create_market_buy_order(s2 + ':USDT', simulated_amount2)   # Close short s2
            else:
                exchange.create_market_buy_order(s1 + ':USDT', simulated_amount1)   # Close short s1
                exchange.create_market_sell_order(s2 + ':USDT', simulated_amount2)  # Close long s2
        
        send_telegram_message(f"Closed position on {current_pair}. Result: {result} {abs(pnl_after_fee):.2f} USD ({pnl_pct:.2f}%). Reason: {exit_reason}.")
        in_position = 0

# Main Trading Loop
def trader_ai_loop():
    global is_trading_active, in_position, current_pair, s1, s2, entry_price1, entry_price2, entry_spread, entry_z, simulated_amount1, simulated_amount2
    best_pair = find_best_pair()
    if not best_pair:
        send_telegram_message("No cointegrated pair found. Trading not started.")
        is_trading_active = False
        return
    
    s1, s2 = best_pair.split(' ~ ')
    current_pair = best_pair
    send_telegram_message(f"Trading started on best pair: {best_pair} (Paper Mode: {PAPER_TRADING})")
    
    while is_trading_active:
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
                    simulated_amount1 = (TRADE_AMOUNT_USD / price1) * LEVERAGE
                    simulated_amount2 = (TRADE_AMOUNT_USD / price2) * LEVERAGE
                    if PAPER_TRADING:
                        simulate_order('SELL', s1, simulated_amount1, price1)  # Short s1
                        simulate_order('BUY', s2, simulated_amount2, price2)   # Long s2
                    else:
                        exchange.create_market_sell_order(s1 + ':USDT', simulated_amount1)  # Short s1
                        exchange.create_market_buy_order(s2 + ':USDT', simulated_amount2)   # Long s2
                    send_telegram_message(f"Started trade on {best_pair} with {TRADE_AMOUNT_USD} USD. Direction: Short Spread. Leverage: {LEVERAGE}x. (Simulated: {PAPER_TRADING})")
                
                elif current_z < -Z_ENTRY:  # Long spread
                    in_position = 1
                    entry_price1, entry_price2 = price1, price2
                    entry_spread = spread[-1]
                    entry_z = current_z
                    simulated_amount1 = (TRADE_AMOUNT_USD / price1) * LEVERAGE
                    simulated_amount2 = (TRADE_AMOUNT_USD / price2) * LEVERAGE
                    if PAPER_TRADING:
                        simulate_order('BUY', s1, simulated_amount1, price1)   # Long s1
                        simulate_order('SELL', s2, simulated_amount2, price2)  # Short s2
                    else:
                        exchange.create_market_buy_order(s1 + ':USDT', simulated_amount1)   # Long s1
                        exchange.create_market_sell_order(s2 + ':USDT', simulated_amount2)  # Short s2
                    send_telegram_message(f"Started trade on {best_pair} with {TRADE_AMOUNT_USD} USD. Direction: Long Spread. Leverage: {LEVERAGE}x. (Simulated: {PAPER_TRADING})")
            
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
                    if PAPER_TRADING:
                        simulate_order('CLOSE', s1, simulated_amount1, price1)
                        simulate_order('CLOSE', s2, simulated_amount2, price2)
                    else:
                        if in_position == 1:
                            exchange.create_market_sell_order(s1 + ':USDT', simulated_amount1)  # Close long s1
                            exchange.create_market_buy_order(s2 + ':USDT', simulated_amount2)   # Close short s2
                        else:
                            exchange.create_market_buy_order(s1 + ':USDT', simulated_amount1)   # Close short s1
                            exchange.create_market_sell_order(s2 + ':USDT', simulated_amount2)  # Close long s2
                    
                    pnl_after_fee = raw_pnl - (entry_price1 + entry_price2) * FEE * LEVERAGE
                    pnl_pct = (pnl_after_fee / TRADE_AMOUNT_USD) * 100
                    result = 'Gain' if pnl_after_fee > 0 else 'Loss'
                    send_telegram_message(f"Ended trade on {best_pair}. Result: {result} {abs(pnl_after_fee):.2f} USD ({pnl_pct:.2f}%). Reason: {exit_reason}. (Simulated: {PAPER_TRADING})")
                    in_position = 0
            
            time.sleep(60 * 60 * 4)  # Sleep 4 hours (match TIMEFRAME)
        
        except Exception as e:
            print(f"Error in trading loop: {e}")
            send_telegram_message(f"Error in Trader AI: {e}")
            time.sleep(60)  # Retry after 1 min

# Telegram Handlers (unchanged from previous)
def start_trading(update, context):
    global is_trading_active, trading_thread
    if str(update.message.chat_id) != TELEGRAM_CHAT_ID:
        update.message.reply_text("Unauthorized user!")
        return
    if not is_trading_active:
        is_trading_active = True
        trading_thread = threading.Thread(target=trader_ai_loop)
        trading_thread.start()
        update.message.reply_text(f"Trading started! (Paper Mode: {PAPER_TRADING})", reply_markup=get_keyboard())
    else:
        update.message.reply_text("Trading is already active.", reply_markup=get_keyboard())

def stop_trading(update, context):
    global is_trading_active
    if str(update.message.chat_id) != TELEGRAM_CHAT_ID:
        update.message.reply_text("Unauthorized user!")
        return
    if is_trading_active:
        close_position('MANUAL_STOP')  # Safely close any open position (simulated if paper mode)
        is_trading_active = False
        update.message.reply_text("Trading stopped.", reply_markup=get_keyboard())
    else:
        update.message.reply_text("Trading is not active.", reply_markup=get_keyboard())

def button_handler(update, context):
    text = update.message.text
    if text == "Start Trading":
        start_trading(update, context)
    elif text == "Stop Trading":
        stop_trading(update, context)

# Persistent Keyboard with Buttons (unchanged)
def get_keyboard():
    from telegram import ReplyKeyboardMarkup
    keyboard = [["Start Trading", "Stop Trading"]]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)  # Persistent

# Welcome Command (unchanged)
def welcome(update, context):
    if str(update.message.chat_id) != TELEGRAM_CHAT_ID:
        update.message.reply_text("Unauthorized user!")
        return
    update.message.reply_text(f"Welcome to Trader AI! Use the buttons to control trading. (Paper Mode: {PAPER_TRADING})", reply_markup=get_keyboard())

# Main Bot Setup (unchanged)
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", welcome))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, button_handler))
    
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
