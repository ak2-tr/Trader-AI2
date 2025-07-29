#!/usr/bin/env python3
import os, sys, time, asyncio, traceback, requests, warnings, csv
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes

# ——— ENVIRONMENT ———
os.environ["TZ"] = "UTC"
warnings.filterwarnings("ignore")

# ——— CONFIGURATION ———
WALLEX_API_KEY     = os.getenv("WALLEX_API_KEY", "")
BASE_URL           = "https://api.wallex.ir/v1"
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOLS            = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT",
                      "ARBUSDT","TRXUSDT","DOGEUSDT","ADAUSDT","LINKUSDT"]
TIMEFRAME          = "4h"
LOOKBACK           = 300
COINTEG_THRESHOLD  = 0.05
WINDOW             = 50
Z_ENTRY            = 2.0
Z_EXIT             = 0.5
STOPLOSS_PCT       = 0.04   # 4% adverse move
STOPLOSS_Z         = 3.0    # |Δz| > 3
LEVERAGE           = 10
FEE                = 0.001  # 0.1% per leg
PAPER_TRADING      = True

# اگر پس از این ساعات سیگنال ورود نداشتیم، جفت را ری‌اسکن کن
RESCAN_HOURS       = 12

# ——— WALLET / MONEY MANAGEMENT ———
INITIAL_WALLET       = 100.0   # $100 start
DAILY_ALLOC_FRAC     = 0.5     # 50% of wallet per trade
wallet_balance       = INITIAL_WALLET
n_open_positions     = 0

# ——— GLOBAL TRADING STATE ———
is_trading_active = False
trading_task      = None

# Per‐trade state
in_position     = 0   # 0=no, +1=long‐spread, -1=short‐spread
current_pair    = None
beta_ratio      = 1.0  # hedge‐ratio
entry_price1    = 0.0
entry_price2    = 0.0
entry_spread    = 0.0
entry_zscore    = 0.0
qty1, qty2      = 0.0, 0.0
pair_found_time = None

# فایل لاگ معاملات (CSV)
LOG_FILE = "trade_log.csv"
FIELDNAMES = ["time","pair","side","price1_entry","price2_entry",
              "price1_exit","price2_exit","pnl","reason"]

def init_trade_log():
    """اگر فایل لاگ وجود ندارد، هدر بنویس."""
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def save_trade_log(rec: dict):
    """ضبط رکورد معامله به CSV."""
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(rec)

# ——— WALLEX CLIENT ———
class WallexClient:
    def __init__(self, api_key: str):
        self.api_key     = api_key
        self.MARKET_BASE = BASE_URL
        self._kline_eps  = ["/market/kline", "/market/candles"]

    def _hdrs(self):
        return {"x-api-key": self.api_key}

    def _interval_to_seconds(self, inter: str) -> int:
        if inter.endswith("m"): return int(inter[:-1])*60
        if inter.endswith("h"): return int(inter[:-1])*3600
        if inter.endswith("d"): return int(inter[:-1])*86400
        raise ValueError(inter)

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int=100):
        """دریافت OHLCV با اولویت ep اول، در صورت خطا ep دوم."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        last_err = None
        for ep in self._kline_eps:
            try:
                r = requests.get(self.MARKET_BASE+ep, headers=self._hdrs(),
                                 params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json().get("result") or r.json().get("data") or []
                    ohlcv = []
                    for b in data:
                        ohlcv.append([
                            int(b["t"]), float(b["o"]), float(b["h"]),
                            float(b["l"]), float(b["c"]), float(b["v"])
                        ])
                    if ohlcv: return ohlcv
                last_err = f"{ep} → {r.status_code}"
            except Exception as e:
                last_err = str(e)
        # fallback به udf/history
        try:
            now_sec = int(time.time())
            span    = self._interval_to_seconds(interval) * limit
            params2 = {
                "symbol": symbol,
                "resolution": interval.rstrip("hmd"),
                "from": now_sec-span, "to": now_sec
            }
            r2 = requests.get(BASE_URL+"/udf/history", headers=self._hdrs(),
                              params=params2, timeout=10)
            j2 = r2.json()
            d = j2.get("result") or j2.get("value") or j2
            ts, o, h, l, c, v = d["t"], d["o"], d["h"], d["l"], d["c"], d["v"]
            return [[int(ts[i])*1000, o[i], h[i], l[i], c[i], v[i]] for i in range(len(ts))]
        except Exception as e:
            raise RuntimeError(f"fetch_ohlcv failed: {last_err} | {e}")

    def create_market_order(self, symbol: str, side: str, amount: float):
        return requests.post(
            f"{self.MARKET_BASE}/order",
            headers=self._hdrs(),
            json={"symbol": symbol, "side": side, "amount": amount},
            timeout=10
        )

exchange = WallexClient(WALLEX_API_KEY)

# ——— TELEGRAM ———
bot = Bot(token=TELEGRAM_TOKEN)
async def send_message(text: str, reply_markup=None):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                           text=text, parse_mode=ParseMode.HTML,
                           reply_markup=reply_markup)

def build_menu():
    btns = [[
        InlineKeyboardButton("▶️ Start", callback_data="start"),
        InlineKeyboardButton("⏹️ Stop", callback_data="stop")
    ]]
    return InlineKeyboardMarkup(btns)

# ——— DATA PREP ———
def fetch_live_data(sym: str) -> pd.DataFrame:
    try:
        ohlcv = exchange.fetch_ohlcv(sym, TIMEFRAME, LOOKBACK)
        df = pd.DataFrame(ohlcv, columns=["datetime","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["close"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print("fetch_live_data error:", e)
        return pd.DataFrame()

def find_best_pair():
    """جفتی را برمی‌گرداند که کمترین p‐value < آستانه دارد."""
    data = {s: fetch_live_data(s) for s in SYMBOLS}
    best, best_p = None, 1.0
    for i in range(len(SYMBOLS)):
        for j in range(i+1, len(SYMBOLS)):
            a, b = SYMBOLS[i], SYMBOLS[j]
            df1, df2 = data[a], data[b]
            if df1.empty or df2.empty: continue
            dfm = pd.merge(df1[["datetime","close"]],
                           df2[["datetime","close"]],
                           on="datetime").dropna()
            if len(dfm) < WINDOW: continue
            _, pval, _ = coint(dfm["close_x"], dfm["close_y"])
            if pval < COINTEG_THRESHOLD and pval < best_p:
                best_p, best = pval, f"{a}~{b}"
    return best

# ——— UTILITY ———
async def smart_sleep(total_sec: int):
    """به‌جای یک‌دفعه sleep، هر ۶۰ ثانیه توقف را چک می‌کند."""
    chunks = max(1, total_sec // 60)
    for _ in range(chunks):
        if not is_trading_active:
            break
        await asyncio.sleep(60)

# ——— TRADING LOOP ———
async def trading_loop():
    global in_position, current_pair, beta_ratio
    global entry_price1, entry_price2, entry_spread, entry_zscore, qty1, qty2
    global wallet_balance, n_open_positions, pair_found_time

    # 1) انتخاب جفت اولیه
    best = await asyncio.to_thread(find_best_pair)
    if not best:
        await send_message("❌ <b>No cointegrated pair found.</b>")
        return
    s1, s2 = best.split("~")
    current_pair = best
    pair_found_time = pd.Timestamp.utcnow()
    await send_message(f"✅ <b>Trading STARTED</b> on <code>{best}</code>\n"
                       f"💰 Wallet: {wallet_balance:.2f} USD")

    # 2) حلقه‌ی اصلی
    while is_trading_active:
        try:
            now = pd.Timestamp.utcnow()

            # ——— اگر پوزیشن باز نداریم و زمان اسکن تمام شده:
            if in_position == 0:
                elapsed = (now - pair_found_time).total_seconds()
                if elapsed > RESCAN_HOURS*3600:
                    await send_message(f"⌛️ {RESCAN_HOURS}h گذشت، اسکن مجدد جفت...")
                    new_best = await asyncio.to_thread(find_best_pair)
                    if new_best and new_best != current_pair:
                        s1, s2 = new_best.split("~")
                        current_pair = new_best
                        pair_found_time = pd.Timestamp.utcnow()
                        await send_message(f"🔄 <b>New pair:</b> {new_best}")
                    else:
                        await smart_sleep(3600)
                        continue

            # ——— دریافت داده و محاسبه زِد-اسکور با hedge‐ratio دینامیک
            df1 = await asyncio.to_thread(fetch_live_data, s1)
            df2 = await asyncio.to_thread(fetch_live_data, s2)
            if df1.empty or df2.empty:
                await smart_sleep(60); continue

            dfm = pd.merge(df1[["datetime","close"]].rename(columns={"close":"c1"}),
                           df2[["datetime","close"]].rename(columns={"close":"c2"}),
                           on="datetime").tail(WINDOW+5)
            if len(dfm) < WINDOW:
                await smart_sleep(60); continue

            p1 = dfm["c1"].values
            p2 = dfm["c2"].values

            # 1) OLS برای β
            beta, intercept = np.polyfit(p2, p1, 1)
            beta_ratio = beta  # ذخیره در متغیر global
            # 2) سری residual
            res = p1 - (beta * p2 + intercept)
            mu = pd.Series(res).rolling(WINDOW).mean().iloc[-1]
            sd = pd.Series(res).rolling(WINDOW).std().iloc[-1] + 1e-8
            z  = (res[-1] - mu) / sd

            # ——— ENTRY ———
            if in_position == 0:
                signal = 0
                if z >  Z_ENTRY: signal = -1
                if z < -Z_ENTRY: signal = +1

                if signal != 0:
                    # تخصیص سرمایه
                    alloc = wallet_balance * DAILY_ALLOC_FRAC
                    alloc_leg = alloc / 2.0
                    price1 = p1[-1]; price2 = p2[-1]
                    qty1   = alloc_leg/price1 * LEVERAGE
                    qty2   = alloc_leg/price2 * LEVERAGE

                    side1 = "BUY"  if signal>0 else "SELL"
                    side2 = "SELL" if signal>0 else "BUY"

                    if PAPER_TRADING:
                        print(f"[SIM ENTRY] {side1} {s1} {qty1:.4f}@{price1:.2f}, "
                              f"{side2} {s2} {qty2:.4f}@{price2:.2f}")
                    else:
                        # اجرای سفارش و چک status_code
                        r1 = exchange.create_market_order(s1, side1, qty1)
                        r2 = exchange.create_market_order(s2, side2, qty2)
                        if r1.status_code!=200 or r2.status_code!=200:
                            await send_message("❗️<b>Order failed!</b>")
                            continue

                    in_position     = signal
                    entry_price1    = price1
                    entry_price2    = price2
                    entry_spread    = res[-1]
                    entry_zscore    = z
                    n_open_positions += 1

                    await send_message(
                        f"🟢 <b>ENTER {'LONG' if signal>0 else 'SHORT'} SPREAD</b>\n"
                        f"pair: {current_pair}  β={beta_ratio:.4f}\n"
                        f"{s1} {side1} {qty1:.4f}@{price1:.2f}\n"
                        f"{s2} {side2} {qty2:.4f}@{price2:.2f}\n"
                        f"z_entry={entry_zscore:.2f}"
                    )
                else:
                    await smart_sleep(60)
                    continue

            # ——— EXIT / STOPLOSS ———
            else:
                # محاسبه PnL
                price1 = p1[-1]; price2 = p2[-1]
                # Δهای هر پا
                delta1 = (price1 - entry_price1) * qty1 * in_position
                delta2 = (entry_price2 - price2) * qty2 * in_position
                gross_pnl = delta1 + delta2
                fee = (qty1*price1 + qty2*price2) * FEE
                net_pnl = gross_pnl - fee

                # شرایط خروج
                exit_reason = None
                pct_move = gross_pnl / (abs(entry_price1-entry_price2)+1e-8)
                if pct_move < -STOPLOSS_PCT:
                    exit_reason = "STOP_PCT"
                elif abs(z-entry_zscore) > STOPLOSS_Z:
                    exit_reason = "STOP_Z"
                elif abs(z) < Z_EXIT:
                    exit_reason = "EXIT_Z"

                if exit_reason:
                    if PAPER_TRADING:
                        print(f"[SIM EXIT] CLOSE {s1} {qty1:.4f}@{price1:.2f}, "
                              f"CLOSE {s2} {qty2:.4f}@{price2:.2f}")
                    else:
                        r1 = exchange.create_market_order(s1,
                                  "SELL" if in_position>0 else "BUY", qty1)
                        r2 = exchange.create_market_order(s2,
                                  "BUY"  if in_position>0 else "SELL", qty2)
                        # چک خطا
                        if r1.status_code!=200 or r2.status_code!=200:
                            await send_message("❗️<b>Order close failed!</b>")
                            continue

                    # بروزرسانی والت
                    wallet_balance += net_pnl
                    in_position     = 0
                    n_open_positions= max(0,n_open_positions-1)

                    # لاگ کردن به CSV
                    rec = {
                        "time": pd.Timestamp.utcnow(),
                        "pair": current_pair,
                        "side": "LONG" if entry_zscore<z else "SHORT",
                        "price1_entry": entry_price1,
                        "price2_entry": entry_price2,
                        "price1_exit": price1,
                        "price2_exit": price2,
                        "pnl": round(net_pnl,4),
                        "reason": exit_reason
                    }
                    save_trade_log(rec)

                    emoji = "✅" if net_pnl>0 else "❌"
                    await send_message(
                        f"{emoji} <b>EXIT {exit_reason}</b>\n"
                        f"P&L: {net_pnl:.2f} USD\n"
                        f"Wallet: {wallet_balance:.2f} USD"
                    )

                else:
                    await smart_sleep(60)

        except Exception as e:
            traceback.print_exc()
            await send_message(f"❗️<b>Error:</b> {e}")
            await smart_sleep(60)

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
                f"Wallet: {wallet_balance:.2f} USD"
            )
            await send_message(msg)
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass

# ——— TELEGRAM CALLBACK ———
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
    init_trade_log()
    await send_message(
        "<b>🤖 Trader Bot v2 is live.</b>\n"
        "▶️ to START  |  ⏹️ to STOP.\n"
        f"PAPER mode: {PAPER_TRADING}\n"
        f"Initial wallet: {INITIAL_WALLET:.2f} USD",
        reply_markup=build_menu()
    )
    asyncio.create_task(heartbeat_loop())
    app = ApplicationBuilder().token(TELEGRAM_TOKEN)\
                              .network_timeout(30).read_timeout(20).build()
    app.add_handler(CallbackQueryHandler(on_callback))
    await app.initialize(); await app.start(); await app.updater.start_polling()
    print("▶️ Bot polling. Ctrl-C to exit.")
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await app.updater._stop_polling(); await app.stop(); await app.shutdown()

if __name__=="__main__":
    if sys.platform.startswith("win") and sys.version_info >= (3,8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
