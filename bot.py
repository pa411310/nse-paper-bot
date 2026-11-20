import os
import io
import time
import requests
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
# UNLIMITED SLOTS MODE (Every valid setup gets tested)
IND_ALLOC_PER_TRADE = 20000.0     # Rs. 20,000 per trade in Mode 1 (Individual Lab)
MASTER_BASE_ALLOC = 50000.0       # Rs. 50,000 per trade in Mode 2 (Normal / Bullish)
MASTER_DEFENSIVE_ALLOC = 25000.0  # Rs. 25,000 per trade in Mode 2 (Cautious / ROC >= 45)
MASTER_COOLDOWN_DAYS = 3          # Wait 3 days after SL before re-entering same stock

# SHARED QUALITY & EXIT RULES
MAX_HOLD_DAYS = 30                # 1-Month Time Stop
MIN_STOCK_PRICE = 50.0            # Ignore penny stocks below Rs. 50
MIN_AVG_TURNOVER = 15000000.0     # Min Rs. 1.5 Cr avg daily traded value
MIN_MCAP_INR = 15000000000.0      # Rs. 1,500 Crore for Mode 2 Fundamental Gate

IND_FILE = "trades_individual.csv"
MASTER_FILE = "trades_master.csv"
REPORT_FILE = "REPORT.md"
NSE_CACHE_FILE = "nse_stocks.csv"

STRATEGIES = [
    "S1_Master_Breakout_VCP",
    "S2_MACD_200EMA_Bounce",
    "S3_Channel_Support_Swing",
    "S4_20EMA_Momentum_Pullback"
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def get_ist_now():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception:
        pass


def get_all_nse_tickers():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200 and "SYMBOL" in r.text:
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip()
            if "SERIES" in df.columns:
                df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
            symbols = [f"{s.strip()}.NS" for s in df["SYMBOL"].dropna().unique()]
            pd.DataFrame({"Symbol": symbols}).to_csv(NSE_CACHE_FILE, index=False)
            print(f"Fetched {len(symbols)} live NSE EQ stocks.")
            return symbols
    except Exception as e:
        print(f"Live NSE fetch warning: {e}")

    if os.path.exists(NSE_CACHE_FILE):
        symbols = pd.read_csv(NSE_CACHE_FILE)["Symbol"].dropna().tolist()
        print(f"Loaded {len(symbols)} stocks from cache.")
        return symbols

    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "ITC.NS", "SBIN.NS", "LT.NS", "BHARTIARTL.NS", "TITAN.NS",
        "SUNPHARMA.NS", "MARUTI.NS", "TATASTEEL.NS", "KPITTECH.NS", "JKIL.NS"
    ]


def check_market_regime():
    try:
        daily = yf.download("^NSEI", period="6mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = [c[0] for c in daily.columns]
        daily = daily.dropna(subset=["Close"])

        monthly = yf.download("^NSEI", period="3y", interval="1mo", progress=False, auto_adjust=True)
        if isinstance(monthly.columns, pd.MultiIndex):
            monthly.columns = [c[0] for c in monthly.columns]
        monthly = monthly.dropna(subset=["Close"])

        roc_18m = None
        if len(monthly) >= 19:
            c_now = float(monthly["Close"].iloc[-1])
            c_18m = float(monthly["Close"].iloc[-19])
            roc_18m = ((c_now - c_18m) / c_18m) * 100

        if len(daily) < 55:
            return True, MASTER_BASE_ALLOC, "NORMAL (Default)"

        close = float(daily["Close"].iloc[-1])
        prev_close = float(daily["Close"].iloc[-2])
        ema50 = float(daily["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        day_chg = ((close - prev_close) / prev_close) * 100
        roc_txt = f"18M ROC: {roc_18m:.1f}%" if roc_18m is not None else "18M ROC: N/A"

        if day_chg <= -2.0:
            return False, 0.0, f"CRASH PAUSE (Nifty {day_chg:.2f}% today | {roc_txt})"
        if roc_18m is not None and roc_18m >= 45.0:
            return True, MASTER_DEFENSIVE_ALLOC, f"OVERHEATED THROTTLE (50% Size | {roc_txt})"
        if close < ema50:
            return True, MASTER_DEFENSIVE_ALLOC, f"CAUTIOUS THROTTLE (Nifty < 50 EMA, 50% Size | {roc_txt})"
        if roc_18m is not None and roc_18m <= 5.0:
            return True, MASTER_BASE_ALLOC, f"BOTTOM ZONE AGGRESSIVE (100% Size | {roc_txt})"

        return True, MASTER_BASE_ALLOC, f"BULLISH NORMAL (100% Size | Nifty > 50 EMA | {roc_txt})"
    except Exception:
        return True, MASTER_BASE_ALLOC, "NORMAL (Fallback)"


def check_fundamentals_smart(stock):
    """
    Checks MCap (>= Rs. 1,500 Cr), PE, ROE, Debt/Equity with rate-limit pause
    and fast_info backup so microcaps cannot sneak through as blank 'Pass'.
    """
    time.sleep(0.5)
    try:
        t = yf.Ticker(stock)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        mcap = info.get("marketCap")
        if mcap is None:
            try:
                fi = t.fast_info
                mcap = fi["marketCap"] if "marketCap" in fi else getattr(fi, "market_cap", None)
            except Exception:
                mcap = None

        if mcap is None:
            return False, 0.0, "NoMCapData"
        if mcap < MIN_MCAP_INR:
            return False, 0.0, f"MCap<{int(mcap/1e7)}Cr"

        pe = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        de_raw = info.get("debtToEquity")
        inst = info.get("heldPercentInstitutions")

        if pe is not None and (pe <= 0 or pe > 65):
            return False, 0.0, f"PE={pe:.1f}"

        de = None
        if de_raw is not None:
            de = (de_raw / 100.0) if de_raw > 5 else de_raw
            if de > 2.0:
                return False, 0.0, f"DE={de:.2f}"

        if roe is not None and roe < 0:
            return False, 0.0, "NegROE"

        bonus = 0.0
        tags = [f"MC:{int(mcap/1e7)}Cr"]
        if pe is not None and 10 <= pe <= 45:
            bonus += 1.0
            tags.append(f"PE:{pe:.0f}")
        if roe is not None and roe >= 0.15:
            bonus += 1.5
            tags.append(f"ROE:{roe*100:.0f}%")
        if de is not None and de <= 0.8:
            bonus += 1.0
            tags.append(f"DE:{de:.2f}")
        if inst is not None and inst >= 0.10:
            bonus += 1.0
            tags.append(f"Inst:{inst*100:.0f}%")

        return True, bonus, ",".join(tags)
    except Exception:
        return False, 0.0, "API-Error"


def load_trades(filepath):
    cols = [
        "trade_id", "strategy", "stock", "status", "buy_date", "buy_price",
        "qty", "stop_loss", "target", "sell_date", "sell_price",
        "exit_reason", "last_price", "pnl", "pnl_pct", "notes"
    ]
    str_cols = ["strategy", "stock", "status", "buy_date", "sell_date", "exit_reason", "notes"]
    float_cols = ["buy_price", "stop_loss", "target", "sell_price", "last_price", "pnl", "pnl_pct"]

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, dtype={c: object for c in str_cols})
        for c in cols:
            if c not in df.columns:
                df[c] = "" if c in str_cols else np.nan
        for c in str_cols:
            df[c] = df[c].astype(object).fillna("")
        for c in float_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return df[cols]

    df = pd.DataFrame(columns=cols)
    for c in str_cols:
        df[c] = df[c].astype(object)
    for c in float_cols:
        df[c] = df[c].astype(float)
    return df


def append_trade(trades_df, new_row):
    row_df = pd.DataFrame([new_row])
    if trades_df.empty:
        return row_df
    return pd.concat([trades_df, row_df], ignore_index=True)


def compute_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    df["Vol_30MA"] = vol.rolling(30).mean()
    df["Turnover_20MA"] = (close * vol).rolling(20).mean()
    return df


def download_batch_data(tickers, batch_size=100):
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(
                batch, period="1y", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True
            )
            if raw.empty:
                continue

            if len(batch) == 1:
                t = batch[0]
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw[t] if t in raw.columns.get_level_values(0) else raw.droplevel(1, axis=1)
                else:
                    df = raw
                df = df.dropna(subset=["Close", "High", "Low", "Open", "Volume"])
                if len(df) >= 210:
                    all_data[t] = df
            else:
                avail = set(raw.columns.get_level_values(0))
                for t in batch:
                    if t in avail:
                        df = raw[t].dropna(subset=["Close", "High", "Low", "Open", "Volume"])
                        if len(df) >= 210:
                            all_data[t] = df
        except Exception:
            continue
        time.sleep(0.4)
    return all_data


def manage_open_trades(trades_df, market_data, today_str):
    closed_today = []
    if trades_df.empty:
        return trades_df, closed_today

    today_dt = datetime.strptime(today_str, "%Y-%m-%d")

    for idx, row in trades_df[trades_df["status"] == "OPEN"].iterrows():
        stock = row["stock"]
        if stock not in market_data:
            continue

        df = compute_indicators(market_data[stock])
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        c_high = float(latest["High"])
        c_low = float(latest["Low"])
        c_close = float(latest["Close"])

        buy_price = float(row["buy_price"])
        qty = int(row["qty"])
        sl = float(row["stop_loss"])
        target = float(row["target"])
        buy_dt = datetime.strptime(str(row["buy_date"]), "%Y-%m-%d")
        days_held = (today_dt - buy_dt).days

        sell_price = None
        reason = None

        check_low = c_close if days_held == 0 else c_low
        check_high = c_close if days_held == 0 else c_high

        if check_low <= sl:
            sell_price = c_close if days_held == 0 else sl
            reason = "SL_HIT"
        elif check_high >= target:
            sell_price = c_close if days_held == 0 else target
            reason = "TARGET_HIT"
        elif (
            "S1" in str(row["strategy"])
            and days_held >= 3
            and c_close < float(latest["EMA_21"])
            and float(prev["Close"]) < float(prev["EMA_21"])
            and c_close < float(latest["Open"])
            and float(prev["Close"]) < float(prev["Open"])
        ):
            sell_price = c_close
            reason = "21EMA_TRAIL_EXIT"
        elif days_held >= MAX_HOLD_DAYS:
            sell_price = c_close
            reason = "TIME_STOP_30D"

        if sell_price is not None:
            pnl = (sell_price - buy_price) * qty
            pnl_pct = ((sell_price - buy_price) / buy_price) * 100
            trades_df.at[idx, "status"] = "CLOSED"
            trades_df.at[idx, "sell_date"] = today_str
            trades_df.at[idx, "sell_price"] = round(sell_price, 2)
            trades_df.at[idx, "exit_reason"] = reason
            trades_df.at[idx, "last_price"] = round(c_close, 2)
            trades_df.at[idx, "pnl"] = round(pnl, 2)
            trades_df.at[idx, "pnl_pct"] = round(pnl_pct, 2)
            closed_today.append(
                f"SELL {stock} ({row['strategy']}) @ {sell_price:.2f} | {reason} | Rs.{pnl:+.0f} ({pnl_pct:+.1f}%)"
            )
        else:
            u_pnl = (c_close - buy_price) * qty
            u_pct = ((c_close - buy_price) / buy_price) * 100
            trades_df.at[idx, "last_price"] = round(c_close, 2)
            trades_df.at[idx, "pnl"] = round(u_pnl, 2)
            trades_df.at[idx, "pnl_pct"] = round(u_pct, 2)

    return trades_df, closed_today


def scan_all_signals(market_data):
    candidates = {s: [] for s in STRATEGIES}

    for stock, raw_df in market_data.items():
        try:
            df = compute_indicators(raw_df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            close = float(latest["Close"])
            open_p = float(latest["Open"])
            high = float(latest["High"])
            low = float(latest["Low"])
            vol = float(latest["Volume"])
            vol_30ma = float(latest["Vol_30MA"])
            turnover = float(latest["Turnover_20MA"])
            atr = float(latest["ATR_14"])

            if close < MIN_STOCK_PRICE or turnover < MIN_AVG_TURNOVER or np.isnan(atr) or atr <= 0 or vol_30ma <= 0:
                continue

            close_20d_ago = float(df["Close"].iloc[-21])
            ret_1m_pct = ((close - close_20d_ago) / close_20d_ago) * 100
            prev_close = float(prev["Close"])
            day_gain_pct = ((close - prev_close) / prev_close) * 100
            max_15d_drop = float(((df["Close"].tail(15) - df["Open"].tail(15)) / df["Open"].tail(15)).min()) * 100

            not_overextended = (ret_1m_pct <= 25.0) and (day_gain_pct <= 9.0)
            clean_behavior = not_overextended and (max_15d_drop >= -7.5)

            ema20 = float(latest["EMA_20"])
            ema50 = float(latest["EMA_50"])
            ema200 = float(latest["EMA_200"])

            # S1: Progressive VCP + Prior Impulse + Volume Surge
            prior_15 = df.iloc[-16:-1]
            prior_5 = df.iloc[-6:-1]
            prior_60 = df.iloc[-61:-1]

            range_15 = (float(prior_15["High"].max()) - float(prior_15["Low"].min())) / float(prior_15["Low"].min())
            range_5 = (float(prior_5["High"].max()) - float(prior_5["Low"].min())) / float(prior_5["Low"].min())
            prior_impulse = (float(prior_60["High"].max()) - float(prior_60["Low"].min())) / float(prior_60["Low"].min())
            prior_vol_calm = float(prior_5["Volume"].mean()) <= 1.15 * vol_30ma

            if (
                close > ema50 > ema200
                and prior_impulse >= 0.15
                and range_15 <= 0.16
                and range_5 < range_15
                and prior_vol_calm
                and close > float(df["High"].iloc[-11:-1].max())
                and vol >= 1.5 * vol_30ma
                and close > open_p
                and not_overextended
            ):
                sl = max(close - (2.0 * atr), close * 0.90)
                target = close + (4.0 * atr)
                score = (vol / vol_30ma) + (1.0 - range_5)
                candidates["S1_Master_Breakout_VCP"].append({
                    "score": score, "stock": stock, "price": close,
                    "sl": sl, "target": target, "clean": clean_behavior
                })

            # S2: MACD + 200 EMA Support Bounce
            macd = float(latest["MACD"])
            macdh = float(latest["MACD_HIST"])
            prev_macdh = float(prev["MACD_HIST"])
            low_20 = float(df["Low"].tail(20).min())

            if (
                close > ema200
                and macd < 0
                and prev_macdh <= 0 < macdh
                and close <= low_20 * 1.08
                and close > open_p
            ):
                sl = max(ema200 * 0.99, close * 0.92)
                risk = close - sl
                if risk > 0:
                    target = close + (1.5 * risk)
                    candidates["S2_MACD_200EMA_Bounce"].append({
                        "score": abs(macd), "stock": stock, "price": close,
                        "sl": sl, "target": target, "clean": clean_behavior
                    })

            # S3: Trend Line Channel Support Swing
            high_20 = float(df["High"].tail(20).max())
            channel_width = (high_20 - low_20) / low_20 if low_20 > 0 else 0
            candle_range = high - low

            if (
                close > ema200
                and 0.08 <= channel_width <= 0.30
                and low <= low_20 * 1.02
                and close > open_p
                and candle_range > 0
                and ((close - low) / candle_range) >= 0.5
            ):
                sl = low_20 * 0.98
                target = high_20 * 0.99
                if target > close * 1.05 and close > sl:
                    candidates["S3_Channel_Support_Swing"].append({
                        "score": channel_width * 10, "stock": stock, "price": close,
                        "sl": sl, "target": target, "clean": clean_behavior
                    })

            # S4: 20-EMA Momentum Pullback (With 1.2x avg volume quality filter)
            if (
                close > ema50 > ema200
                and low <= ema20 * 1.015
                and close >= ema20
                and close > open_p
                and vol >= float(prev["Volume"])
                and vol >= 1.2 * vol_30ma
                and not_overextended
            ):
                sl = max(low * 0.99, close * 0.94)
                risk = close - sl
                if risk > 0:
                    target = close + (3.0 * risk)
                    candidates["S4_20EMA_Momentum_Pullback"].append({
                        "score": vol / vol_30ma, "stock": stock, "price": close,
                        "sl": sl, "target": target, "clean": clean_behavior
                    })

        except Exception:
            continue

    return candidates


def get_blocked_stocks(trades_df, today_str, cooldown_days, strategy_filter=None):
    if trades_df.empty:
        return set()
    today_dt = datetime.strptime(today_str, "%Y-%m-%d")
    blocked = set()

    df = trades_df[trades_df["status"] == "CLOSED"]
    if strategy_filter:
        df = df[df["strategy"] == strategy_filter]

    for _, r in df.iterrows():
        s_date = str(r["sell_date"]).strip()
        if not s_date or s_date.lower() == "nan":
            continue
        diff = (today_dt - datetime.strptime(s_date, "%Y-%m-%d")).days
        if diff == 0 or (r["exit_reason"] == "SL_HIT" and diff <= cooldown_days):
            blocked.add(r["stock"])
    return blocked


def execute_mode1_individual(trades_df, candidates, today_str):
    """MODE 1: UNLIMITED trades across all 4 strategies independently."""
    bought = []
    next_id = int(trades_df["trade_id"].max()) + 1 if not trades_df.empty else 1

    for strat in STRATEGIES:
        open_df = trades_df[(trades_df["strategy"] == strat) & (trades_df["status"] == "OPEN")]
        open_stocks = set(open_df["stock"].tolist())
        blocked_today = get_blocked_stocks(trades_df, today_str, cooldown_days=0, strategy_filter=strat)

        ranked = sorted(candidates[strat], key=lambda x: x["score"], reverse=True)
        for item in ranked:
            stock = item["stock"]
            if stock in open_stocks or stock in blocked_today:
                continue

            qty = int(IND_ALLOC_PER_TRADE // item["price"])
            if qty < 1:
                continue

            new_row = {
                "trade_id": next_id, "strategy": strat, "stock": stock, "status": "OPEN",
                "buy_date": today_str, "buy_price": round(item["price"], 2), "qty": qty,
                "stop_loss": round(item["sl"], 2), "target": round(item["target"], 2),
                "sell_date": "", "sell_price": np.nan, "exit_reason": "",
                "last_price": round(item["price"], 2), "pnl": 0.0, "pnl_pct": 0.0,
                "notes": "TechPure"
            }
            trades_df = append_trade(trades_df, new_row)
            open_stocks.add(stock)
            next_id += 1
            bought.append(f"[IND] BUY {stock} ({strat}) @ {item['price']:.2f}")

    return trades_df, bought


def execute_mode2_master(trades_df, candidates, today_str, allow_buys, alloc_amt):
    """MODE 2: UNLIMITED smart buys (No cross-strategy duplicates + Strict Fundamentals + Confluence)."""
    bought = []
    funnel = {"raw_unique": 0, "clean_passed": 0, "fund_passed": 0, "bought": 0}

    if not allow_buys:
        return trades_df, bought, funnel

    open_df = trades_df[trades_df["status"] == "OPEN"]
    open_stocks = set(open_df["stock"].tolist())
    cooldown_stocks = get_blocked_stocks(trades_df, today_str, cooldown_days=MASTER_COOLDOWN_DAYS)

    stock_pool = {}
    for strat in STRATEGIES:
        for item in candidates[strat]:
            stk = item["stock"]
            if stk not in stock_pool:
                stock_pool[stk] = {
                    "stock": stk, "strategies": [strat], "price": item["price"],
                    "sl": item["sl"], "target": item["target"],
                    "score": item["score"], "clean": item["clean"]
                }
            else:
                stock_pool[stk]["strategies"].append(strat)
                stock_pool[stk]["sl"] = max(stock_pool[stk]["sl"], item["sl"])
                stock_pool[stk]["target"] = max(stock_pool[stk]["target"], item["target"])
                stock_pool[stk]["score"] += item["score"]
                stock_pool[stk]["clean"] = stock_pool[stk]["clean"] and item["clean"]

    funnel["raw_unique"] = len(stock_pool)

    eligible = [
        v for v in stock_pool.values()
        if v["clean"] and v["stock"] not in open_stocks and v["stock"] not in cooldown_stocks
    ]
    funnel["clean_passed"] = len(eligible)

    if not eligible:
        return trades_df, bought, funnel

    eligible.sort(key=lambda x: (len(x["strategies"]), x["score"]), reverse=True)

    fund_approved = []
    for item in eligible[:60]:
        ok, f_bonus, f_tag = check_fundamentals_smart(item["stock"])
        if ok:
            item["final_score"] = item["score"] + f_bonus + (len(item["strategies"]) * 2.0)
            item["f_tag"] = f_tag
            fund_approved.append(item)

    funnel["fund_passed"] = len(fund_approved)
    fund_approved.sort(key=lambda x: x["final_score"], reverse=True)

    next_id = int(trades_df["trade_id"].max()) + 1 if not trades_df.empty else 1

    for item in fund_approved:
        stock = item["stock"]
        qty = int(alloc_amt // item["price"])
        if qty < 1:
            continue

        strat_label = (
            "CONFLUENCE:" + "+".join([s.split("_")[0] for s in item["strategies"]])
            if len(item["strategies"]) > 1
            else item["strategies"][0]
        )

        new_row = {
            "trade_id": next_id, "strategy": strat_label, "stock": stock, "status": "OPEN",
            "buy_date": today_str, "buy_price": round(item["price"], 2), "qty": qty,
            "stop_loss": round(item["sl"], 2), "target": round(item["target"], 2),
            "sell_date": "", "sell_price": np.nan, "exit_reason": "",
            "last_price": round(item["price"], 2), "pnl": 0.0, "pnl_pct": 0.0,
            "notes": item["f_tag"]
        }
        trades_df = append_trade(trades_df, new_row)
        open_stocks.add(stock)
        next_id += 1
        funnel["bought"] += 1
        bought.append(f"[MASTER] BUY {stock} ({strat_label} | {item['f_tag']}) @ {item['price']:.2f}")

    return trades_df, bought, funnel


def calc_stats(df):
    if df.empty:
        return 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    closed = df[df["status"] == "CLOSED"]
    opened = df[df["status"] == "OPEN"]
    wins = len(closed[closed["pnl"] > 0])
    losses = len(closed[closed["pnl"] <= 0])
    wr = (wins / len(closed) * 100) if len(closed) > 0 else 0.0
    r_pnl = float(closed["pnl"].sum()) if len(closed) > 0 else 0.0
    u_pnl = float(opened["pnl"].sum()) if len(opened) > 0 else 0.0
    tot = r_pnl + u_pnl
    deployed = float((df["buy_price"] * df["qty"]).sum())
    roi_pct = (tot / deployed * 100) if deployed > 0 else 0.0
    return len(df), len(opened), len(closed), wins, losses, wr, deployed, r_pnl, u_pnl, tot, roi_pct


def generate_report(ind_df, master_df, ts_str, total_scanned, regime_msg, candidates, funnel):
    raw_counts = " | ".join([f"{s.split('_')[0]}: {len(candidates[s])}" for s in STRATEGIES])
    lines = [
        "# Dual-Mode All-NSE Paper Trading Report (v3.1 - Unlimited Slots)",
        f"**Last Updated (IST):** {ts_str} | **Stocks Scanned:** {total_scanned} NSE Equities",
        f"**Market Regime (Nifty 50 EMA + 18M ROC):** `{regime_msg}`",
        f"**Today's Signal Funnel:** Raw Signals ({raw_counts}) -> Master Clean: {funnel['clean_passed']} -> Fundamental Passed: {funnel['fund_passed']} -> Master Bought: {funnel['bought']}\n",
        "---",
        "## MODE 1: Individual 4 Strategies (Unlimited Slots - Rs. 20,000/Trade)",
        "| Strategy | Total | Open | Closed | Wins | Losses | Win Rate | Deployed (Rs.) | Realized (Rs.) | Unrealized (Rs.) | Net ROI % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for strat in STRATEGIES:
        sdf = ind_df[ind_df["strategy"] == strat]
        t, o, c, w, l, wr, dep, rp, up, tot, roi = calc_stats(sdf)
        lines.append(
            f"| **{strat}** | {t} | {o} | {c} | {w} | {l} | {wr:.1f}% | {dep:,.0f} | {rp:+,.0f} | {up:+,.0f} | **{roi:+.2f}%** |"
        )

    it, io_, ic, iw, il, iwr, idep, irp, iup, itot, iroi = calc_stats(ind_df)
    lines.append(f"\n**Mode 1 Total:** Deployed Rs. {idep:,.0f} | Net P&L: Rs. {itot:+,.0f} (**{iroi:+.2f}% ROI**)\n")

    lines.append("---")
    lines.append("## MODE 2: Master Smart Bot (Unlimited Slots + Strict Fundamentals + No Duplicates)")
    lines.append("| Portfolio | Total | Open | Closed | Wins | Losses | Win Rate | Deployed (Rs.) | Realized (Rs.) | Unrealized (Rs.) | Net ROI % |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    mt, mo, mc, mw, ml, mwr, mdep, mrp, mup, mtot, mroi = calc_stats(master_df)
    lines.append(
        f"| **MASTER SMART BOT** | {mt} | {mo} | {mc} | {mw} | {ml} | {mwr:.1f}% | {mdep:,.0f} | {mrp:+,.0f} | {mup:+,.0f} | **{mroi:+.2f}%** |\n"
    )

    lines.append("### Mode 2 Active Positions (Master Bot)\n")
    m_open = master_df[master_df["status"] == "OPEN"]
    if m_open.empty:
        lines.append("_No active Master positions._\n")
    else:
        lines.append("| Date | Trigger | Stock | Fundamentals | Buy | Current | SL | Target | P&L |")
        lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for _, r in m_open.iterrows():
            lines.append(
                f"| {r['buy_date']} | {r['strategy']} | **{r['stock']}** | {r['notes']} | {r['buy_price']} | {r['last_price']} | {r['stop_loss']} | {r['target']} | {float(r['pnl']):+.0f} ({float(r['pnl_pct']):+.1f}%) |"
            )

    m_closed = master_df[master_df["status"] == "CLOSED"].tail(30)
    lines.append("\n### Mode 2 Closed Trades (Latest 30)\n")
    if m_closed.empty:
        lines.append("_No closed Master trades yet._\n")
    else:
        lines.append("| Buy Date | Sell Date | Trigger | Stock | Buy | Sell | Reason | Net P&L |")
        lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |")
        for _, r in m_closed.iterrows():
            lines.append(
                f"| {r['buy_date']} | {r['sell_date']} | {r['strategy']} | **{r['stock']}** | {r['buy_price']} | {r['sell_price']} | {r['exit_reason']} | {float(r['pnl']):+.0f} ({float(r['pnl_pct']):+.1f}%) |"
            )

    lines.append("\n### Mode 1 Active Positions (Individual Lab)\n")
    i_open = ind_df[ind_df["status"] == "OPEN"]
    if i_open.empty:
        lines.append("_No active Individual positions._\n")
    else:
        lines.append("| Date | Strategy | Stock | Buy | Current | SL | Target | P&L |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for _, r in i_open.iterrows():
            lines.append(
                f"| {r['buy_date']} | {r['strategy']} | **{r['stock']}** | {r['buy_price']} | {r['last_price']} | {r['stop_loss']} | {r['target']} | {float(r['pnl']):+.0f} ({float(r['pnl_pct']):+.1f}%) |"
            )

    i_closed = ind_df[ind_df["status"] == "CLOSED"].tail(30)
    lines.append("\n### Mode 1 Closed Trades (Latest 30)\n")
    if i_closed.empty:
        lines.append("_No closed Individual trades yet._\n")
    else:
        lines.append("| Buy Date | Sell Date | Strategy | Stock | Buy | Sell | Reason | Net P&L |")
        lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |")
        for _, r in i_closed.iterrows():
            lines.append(
                f"| {r['buy_date']} | {r['sell_date']} | {r['strategy']} | **{r['stock']}** | {r['buy_price']} | {r['sell_price']} | {r['exit_reason']} | {float(r['pnl']):+.0f} ({float(r['pnl_pct']):+.1f}%) |"
            )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return iroi, mroi


def main():
    ist_now = get_ist_now()
    today_str = ist_now.strftime("%Y-%m-%d")
    ts_str = ist_now.strftime("%Y-%m-%d %H:%M IST")
    print(f"=== Dual-Mode NSE Paper Bot v3.1 - Unlimited ({ts_str}) ===")

    allow_buys, alloc_amt, regime_msg = check_market_regime()
    print(f"Market Regime: {regime_msg}")

    tickers = get_all_nse_tickers()
    market_data = download_batch_data(tickers, batch_size=100)
    print(f"Valid stocks scanned: {len(market_data)}")

    ind_df = load_trades(IND_FILE)
    master_df = load_trades(MASTER_FILE)

    ind_df, ind_closed = manage_open_trades(ind_df, market_data, today_str)
    master_df, master_closed = manage_open_trades(master_df, market_data, today_str)

    candidates = scan_all_signals(market_data)

    ind_df, ind_bought = execute_mode1_individual(ind_df, candidates, today_str)
    master_df, master_bought, funnel = execute_mode2_master(
        master_df, candidates, today_str, allow_buys, alloc_amt
    )

    ind_df.to_csv(IND_FILE, index=False)
    master_df.to_csv(MASTER_FILE, index=False)
    ind_roi, master_roi = generate_report(
        ind_df, master_df, ts_str, len(market_data), regime_msg, candidates, funnel
    )

    summary = (
        f"NSE Bot v3.1 Unlimited ({ts_str})\n"
        f"Regime: {regime_msg}\n"
        f"Funnel: Raw {funnel['raw_unique']} -> Fund OK {funnel['fund_passed']}\n"
        f"Mode 1 ROI: {ind_roi:+.2f}% (B:{len(ind_bought)} S:{len(ind_closed)})\n"
        f"Mode 2 ROI: {master_roi:+.2f}% (B:{len(master_bought)} S:{len(master_closed)})\n\n"
        + "\n".join(master_bought + master_closed + ind_bought + ind_closed)
    )
    print(summary)
    send_telegram(summary[:4000])


if __name__ == "__main__":
    main()
