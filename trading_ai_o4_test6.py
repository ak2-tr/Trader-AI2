#!/usr/bin/env python3
import os, sys, time, asyncio, traceback, requests, warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes

# ——— ENVIRONMENT & TIMEZONE ———
os.environ["TZ"] = "UTC"
warnings.filterwarnings("ignore")

# ——— CONFIGURATION ———
# Wallex API
WALLEX_API_KEY     = os.getenv("WALLEX_API_KEY", "")
BASE_URL          = "https://api.wallex.ir/v1"
# Telegram
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
# Market & Strategy
SYMBOLS           = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT",
                     "ARBUSDT","TRXUSDT","DOGEUSDT","ADAUSDT","LINKUSDT"]
TIMEFRAME         = "4h"
LOOKBACK          = 300
COINTEG_THRESHOLD = 0.05
WINDOW            = 50
Z_ENTRY           = 2.0
Z_EXIT            = 0.5
STOPLOSS_PCT      = 0.04           # 4% adverse move
STOPLOSS_Z        = 3.0            # |Δz| > 3
LEVERAGE          = 10
FEE               = 0.001          # 0.1% per leg
PAPER_TRADING     = True

RESCAN_HOURS = 12      # اگر پس از ۱۲ ساعت سیگنال ورود نداشتیم، جفت را ری‌اسکن کن


# ——— WALLET / MONEY MANAGEMENT ———
INITIAL_WALLET      = 100.0         # $100 start
DAILY_ALLOC_FRAC    = 0.5           # 50% of wallet per day
# global wallet state (initialized at import)
wallet_balance      = INITIAL_WALLET
daily_alloc_remaining = wallet_balance * DAILY_ALLOC_FRAC
n_open_positions    = 0
today_date          = pd.Timestamp.utcnow().date()

# ——— GLOBAL TRADING STATE ———
is_trading_active = False
trading_task      = None

# Per‐trade state
in_position     = 0      # 0=no, +1=long‐spread, -1=short‐spread
current_pair    = None
s1, s2          = None, None

entry_spread    = 0.0
entry_zscore    = 0.0
qty1, qty2      = 0.0, 0.0

# Trade log (optional, for further analysis)
trade_log = []

# ——— WALLEX HTTP CLIENT ———
class WallexClient:
    def __init__(self, api_key: str):
        self.api_key    = api_key
        self.MARKET_BASE = "https://api.wallex.ir/api/v1"
        self._kline_eps  = ["/market/kline", "/market/candles"]

    def _hdrs(self):
        return {"x-api-key": self.api_key}

    def _interval_to_resolution(self, inter: str) -> str:
        if inter.endswith("m"): return inter[:-1]
        if inter.endswith("h"): return str(int(inter[:-1])*60)
        if inter.endswith("d"): return "D"
        raise ValueError(inter)

    def _interval_to_seconds(self, inter: str) -> int:
        if inter.endswith("m"): return int(inter[:-1])*60
        if inter.endswith("h"): return int(inter[:-1])*3600
        if inter.endswith("d"): return int(inter[:-1])*86400
        raise ValueError(inter)

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int=100):
        # try /market/kline & /market/candles
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        last_err = None
        for ep in self._kline_eps:
            url = self.MARKET_BASE + ep
            try:
                r = requests.get(url, headers=self._hdrs(), params=params, timeout=10)
                if r.status_code==200:
                    payload = r.json().get("result") or r.json().get("data") or []
                    ohlcv = [
                        [int(b["t"]), float(b["o"]), float(b["h"]),
                         float(b["l"]), float(b["c"]), float(b["v"])]
                        for b in payload
                    ]
                    if ohlcv: return ohlcv
                last_err = f"{url} → {r.status_code}"
            except Exception as e:
                last_err = str(e)
        # fallback udf/history
        try:
            resolution = self._interval_to_resolution(interval)
            now_sec    = int(time.time())
            span       = self._interval_to_seconds(interval)*limit
            params2 = {
                "symbol": symbol, "resolution": resolution,
                "from": now_sec-span, "to": now_sec
            }
            r2 = requests.get("https://api.wallex.ir/v1/udf/history",
                              headers=self._hdrs(), params=params2, timeout=10)
            j2 = r2.json()
            data = j2.get("result") or j2.get("value") or j2
            ts, o, h, l, c, v = data["t"], data["o"], data["h"], data["l"], data["c"], data["v"]
            return [[int(ts[i])*1000, o[i], h[i], l[i], c[i], v[i]] for i in range(len(ts))]
        except Exception as e:
            raise RuntimeError(f"fetch_ohlcv failed: {last_err} | {e}")

    def create_market_order(self, symbol: str, side: str, amount: float):
        # real trading
        return requests.post(
            f"{self.MARKET_BASE}/order",
            headers=self._hdrs(),
            json={"symbol": symbol, "side": side, "amount": amount}
        )

exchange = WallexClient(WALLEX_API_KEY)

# ——— TELEGRAM HELPERS ———
bot = Bot(token=TELEGRAM_TOKEN)
async def send_message(text: str, reply_markup: InlineKeyboardMarkup=None):
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

def build_menu() -> InlineKeyboardMarkup:
    btns = [
        [
            InlineKeyboardButton("▶️ Start", callback_data="start"),
            InlineKeyboardButton("⏹️ Stop", callback_data="stop")
        ]
    ]
    return InlineKeyboardMarkup(btns)

# ——— DATA & SIGNALS ———
def fetch_live_data(sym: str) -> pd.DataFrame:
    """
    Fetch OHLCV for `sym` and coerce all columns to the correct types.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(sym, TIMEFRAME, LOOKBACK)
        df = pd.DataFrame(
            ohlcv,
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )

        # 1) Convert timestamp
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

        # 2) Force numeric types on price/volume columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3) (Optional) drop any rows that failed to convert
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:
        print("fetch_live_data error:", e)
        return pd.DataFrame()


def find_best_pair() -> str:
    data = {s: fetch_live_data(s) for s in SYMBOLS}
    best = None; best_p = 1.0
    for i in range(len(SYMBOLS)):
        for j in range(i+1, len(SYMBOLS)):
            a, b = SYMBOLS[i], SYMBOLS[j]
            df1, df2 = data[a], data[b]
            if df1.empty or df2.empty: continue
            dfm = pd.merge(df1[["datetime","close"]], df2[["datetime","close"]],
                            on="datetime").dropna()
            if len(dfm)<WINDOW: continue
            _, pval, _ = coint(dfm["close_x"], dfm["close_y"])
            if pval<COINTEG_THRESHOLD and pval<best_p:
                best_p = pval; best = f"{a}~{b}"
    return best

def simulate_order(action: str, sym: str, amt: float, price: float):
    print(f"[SIM] {action} {sym} {amt:.6f} @ {price:.2f}")

def create_order(side: str, sym: str, amt: float):
    return exchange.create_market_order(sym, side, amt)

# ——— TRADING LOOP ———
async def trading_loop():
    global in_position, s1, s2, current_pair
    global entry_spread, entry_zscore, qty1, qty2
    global wallet_balance, daily_alloc_remaining, n_open_positions, today_date

    # 1) Find & announce best pair
    best = await asyncio.to_thread(find_best_pair)
    if not best:
        await send_message("❌ <b>No cointegrated pair found.</b>")
        return
    s1, s2 = best.split("~")
    current_pair = best
    pair_found_time = pd.Timestamp.utcnow()
    await send_message(
        f"✅ <b>Trading STARTED</b> on <code>{best}</code>\n"
        f"💰 Wallet: {wallet_balance:.2f} USD\n"
        f"📊 Daily alloc: {daily_alloc_remaining:.2f} USD"
    )

    # 2) Main loop
    while is_trading_active:
        try:
            now = pd.Timestamp.utcnow()

                        # ← اینجا: اگر هنوز پوزیشن باز نیستیم و زمان سپری شده از pair_found_time > RESCAN_HOURS
            if in_position == 0:
                elapsed = (now - pair_found_time).total_seconds()
                if elapsed > RESCAN_HOURS * 3600:
                    await send_message(
                        f"⌛️ {RESCAN_HOURS}h گذشت ولی هنوز سیگنال ورود نداشتیم. اسکن مجدد جفت..."
                    )
                    new_best = await asyncio.to_thread(find_best_pair)
                    if new_best and new_best != current_pair:
                        s1, s2 = new_best.split("~")
                        current_pair = new_best
                        pair_found_time = pd.Timestamp.utcnow()
                        await send_message(f"🔄 <b>New pair:</b> {new_best}")
                    else:
                        await asyncio.sleep(3600)
                        continue
            # fetch data
            df1 = await asyncio.to_thread(fetch_live_data, s1)
            df2 = await asyncio.to_thread(fetch_live_data, s2)
            if df1.empty or df2.empty:
                await asyncio.sleep(60); continue

            dfm = pd.merge(
                df1[["datetime","close"]].rename(columns={"close":"c1"}),
                df2[["datetime","close"]].rename(columns={"close":"c2"}),
                on="datetime"
            ).tail(WINDOW+5)
            if len(dfm)<WINDOW:
                await asyncio.sleep(60); continue

            p1 = dfm["c1"].values; p2 = dfm["c2"].values
            spread = p1 - p2
            mu  = pd.Series(spread).rolling(WINDOW).mean().iloc[-1]
            sd  = pd.Series(spread).rolling(WINDOW).std().iloc[-1] + 1e-8
            z   = (spread[-1] - mu)/sd

            # ——— ENTRY ———
            if in_position == 0:
                signal = 0
                if z >  Z_ENTRY: signal = -1
                if z < -Z_ENTRY: signal = +1
                if signal != 0:
                    # allocate
                    alloc_total = daily_alloc_remaining/(n_open_positions+1)
                    if alloc_total <= 0:
                        await send_message("⚠️ Daily allocation exhausted.")
                        break
                    alloc_leg = alloc_total/2.0

                    price1 = p1[-1]; price2 = p2[-1]
                    qty1   = (alloc_leg/price1)*LEVERAGE
                    qty2   = (alloc_leg/price2)*LEVERAGE

                    daily_alloc_remaining -= alloc_total
                    n_open_positions     += 1

                    side1 = "SELL" if signal<0 else "BUY"
                    side2 = "BUY"  if signal<0 else "SELL"

                    if PAPER_TRADING:
                        simulate_order(side1, s1, qty1, price1)
                        simulate_order(side2, s2, qty2, price2)
                    else:
                        await asyncio.to_thread(create_order, side1, s1, qty1)
                        await asyncio.to_thread(create_order, side2, s2, qty2)

                    in_position  = signal
                    entry_spread = spread[-1]
                    entry_zscore = z

                    await send_message(
                        f"🟢 <b>ENTER {'LONG' if signal>0 else 'SHORT'} SPREAD</b>\n"
                        f"{s1} {side1} {qty1:.4f} @ {price1:.2f}\n"
                        f"{s2} {side2} {qty2:.4f} @ {price2:.2f}\n"
                        f"z_entry={entry_zscore:.2f}\n"
                        f"💰 Wallet: {wallet_balance:.2f} USD\n"
                        f"📊 Daily alloc rem: {daily_alloc_remaining:.2f} USD"
                    )
                else:
                    # no entry signal
                    await asyncio.sleep(60)
                    continue

            # ——— EXIT / STOP-LOSS / TRAILING ———
            else:
                curr_spread = p1[-1] - p2[-1]
                raw_pnl     = (curr_spread - entry_spread)*in_position
                pct_move    = raw_pnl/(abs(entry_spread)+1e-8)

                exit_reason = None
                if pct_move < -STOPLOSS_PCT:              exit_reason = "STOP_PCT"
                elif abs(z-entry_zscore) > STOPLOSS_Z:    exit_reason = "STOP_Z"
                elif abs(z) < Z_EXIT:                    exit_reason = "EXIT_Z"

                if exit_reason:
                    price1 = p1[-1]; price2 = p2[-1]
                    side1  = "BUY"  if in_position<0 else "SELL"
                    side2  = "SELL" if in_position<0 else "BUY"

                    if PAPER_TRADING:
                        simulate_order("CLOSE", s1, qty1, price1)
                        simulate_order("CLOSE", s2, qty2, price2)
                    else:
                        await asyncio.to_thread(create_order, side1, s1, qty1)
                        await asyncio.to_thread(create_order, side2, s2, qty2)

                    fee_total = (qty1*price1 + qty2*price2)*FEE
                    net_pnl   = raw_pnl*LEVERAGE - fee_total

                    # update wallet
                    wallet_balance   += net_pnl
                    n_open_positions = max(0, n_open_positions-1)
                    in_position      = 0

                    # log
                    trade_log.append({
                        "pair": current_pair,
                        "entry_spread": entry_spread,
                        "exit_spread": curr_spread,
                        "pnl": net_pnl,
                        "time": pd.Timestamp.utcnow()
                    })

                    emoji = "✅" if net_pnl>0 else "❌"
                    await send_message(
                        f"{emoji} <b>EXIT {exit_reason}</b>\n"
                        f"P&L: {net_pnl:.2f} USD\n"
                        f"💰 Wallet: {wallet_balance:.2f} USD\n"
                        f"📊 Daily alloc rem: {daily_alloc_remaining:.2f} USD"
                    )

            # wait next candle
            await asyncio.sleep(60*60)

        except Exception as e:
            traceback.print_exc()
            await send_message(f"❗️<b>Error:</b> {e}")
            await asyncio.sleep(60)

# ——— HEARTBEAT ———
async def heartbeat_loop():
    try:
        while True:
            status = "🟢 Running" if is_trading_active else "🔴 Stopped"
            msg = (
                f"💓 <b>Heartbeat</b>\n"
                f"Status: {status}\n"
                f"In position: {in_position}\n"
                f"Pair: {current_pair}\n"
                f"💰 Wallet: {wallet_balance:.2f} USD\n"
                f"📊 Daily alloc rem: {daily_alloc_remaining:.2f} USD"
            )
            await send_message(msg)
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass

# ——— TELEGRAM BUTTONS ———
async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global is_trading_active, trading_task
    q = update.callback_query
    await q.answer()
    cmd = q.data

    if cmd=="start":
        if not is_trading_active:
            is_trading_active = True
            trading_task = asyncio.create_task(trading_loop())
            text = "▶️ <b>Trading STARTED</b>"
        else:
            text = "⚠️ Already running."
    elif cmd=="stop":
        if is_trading_active:
            is_trading_active = False
            if trading_task:
                await trading_task
                trading_task = None
            text = "⏹️ <b>Trading STOPPED</b>"
        else:
            text = "⚠️ Already stopped."
    else:
        text = f"❓ Unknown: {cmd}"

    await q.edit_message_text(text=text, parse_mode=ParseMode.HTML,
                              reply_markup=build_menu())

# ——— MAIN ———
async def main():
    # 1) send startup menu
    await send_message(
        "<b>🤖 Trader Bot is live.</b>\n"
        "▶️ to START  |  ⏹️ to STOP.\n"
        f"PAPER mode: {PAPER_TRADING}\n"
        f"Initial wallet: {INITIAL_WALLET:.2f} USD",
        reply_markup=build_menu()
    )
    # 2) heartbeat
    asyncio.create_task(heartbeat_loop())
    # 3) Telegram handlers
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build().network_timeout(30).read_timeout(20)
    app.add_handler(CallbackQueryHandler(on_callback))
    await app.initialize();  await app.start();  await app.updater.start_polling()
    print("▶️ Bot polling. Ctrl-C to exit.")
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await app.updater._stop_polling();  await app.stop();  await app.shutdown()

if __name__=="__main__":
    # Windows selector policy for Py3.8+
    if sys.platform.startswith("win") and sys.version_info >= (3,8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
