"""
NSE Daily Scanner — Automated version
=======================================
Same strategy logic as nse_scanner.py, but:
  - Writes results to results.json (machine-readable, for Telegram/email step)
  - Writes results to docs/index.html (for GitHub Pages, human-readable)
  - Designed to be run headlessly by GitHub Actions on a schedule.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

UNIVERSE = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","LT.NS",
    "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","ITC.NS","BAJFINANCE.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","HCLTECH.NS","WIPRO.NS","ADANIENT.NS",
    "TATASTEEL.NS","TATAMOTORS.NS","BHARTIARTL.NS","ASIANPAINT.NS","HINDUNILVR.NS",
    "NESTLEIND.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","COALINDIA.NS","JSWSTEEL.NS",
    "GRASIM.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS","M&M.NS","BPCL.NS","HINDALCO.NS","TECHM.NS","SBILIFE.NS",
    "HDFCLIFE.NS","BRITANNIA.NS","APOLLOHOSP.NS","INDUSINDBK.NS","UPL.NS",
]

CAPITAL = float(os.environ.get("TRADING_CAPITAL", 100000.0))
RISK_PER_TRADE = 0.01
ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 2.5
COST_ROUNDTRIP = 0.0036

def compute_indicators(df):
    df["EMA20"]  = df["Close"].ewm(span=20).mean()
    df["EMA50"]  = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["ADX"] = dx.rolling(14).mean()
    df["High52w"] = df["Close"].rolling(252, min_periods=50).max()
    df["PctFromHigh"] = (df["Close"] - df["High52w"]) / df["High52w"]
    return df

def scan_stock(ticker):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 220:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = compute_indicators(df).dropna()
    if len(df) < 2:
        return None
    row, prev = df.iloc[-1], df.iloc[-2]
    uptrend = row.EMA20 > row.EMA50 > row.EMA200
    pullback_recovery = prev.RSI < 45 and row.RSI >= 45 and prev.RSI >= 30
    vol_confirm = row.Volume >= 1.3 * row.VolAvg20
    if uptrend and pullback_recovery and vol_confirm:
        entry = row.Close
        stop = entry - ATR_STOP_MULT * row.ATR
        target = entry + ATR_TARGET_MULT * row.ATR
        risk_amt = CAPITAL * RISK_PER_TRADE
        shares = int(risk_amt / (entry - stop)) if entry > stop else 0
        score = row.ADX + (1.3 * row.Volume / row.VolAvg20) + (100 * (1 + row.PctFromHigh))
        return {
            "ticker": ticker, "date": str(df.index[-1].date()),
            "entry": round(float(entry), 2), "stop": round(float(stop), 2),
            "target": round(float(target), 2), "shares": shares,
            "capital_used": round(shares * entry, 0),
            "adx": round(float(row.ADX), 1), "rel_volume": round(float(row.Volume / row.VolAvg20), 2),
            "pct_from_52w_high": round(float(row.PctFromHigh) * 100, 1),
            "score": round(float(score), 1),
            "risk_reward": round((target - entry) / (entry - stop), 2) if entry > stop else None,
        }
    return None

def build_html(hits, run_date):
    rows = ""
    for h in hits:
        rows += f"""<tr>
          <td>{h['ticker']}</td><td>{h['entry']}</td><td>{h['stop']}</td>
          <td>{h['target']}</td><td>{h['risk_reward']}</td>
          <td>{h['shares']}</td><td>{h['adx']}</td><td>{h['rel_volume']}x</td>
        </tr>"""
    if not rows:
        rows = "<tr><td colspan='8' style='text-align:center;padding:20px;'>No qualifying setups today.</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NSE Scanner — {run_date}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width:900px; margin:40px auto; padding:0 20px; background:#0b0f14; color:#e6edf3; }}
h1 {{ font-size:22px; }} .sub {{ color:#9aa5b1; margin-bottom:24px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid #1f2937; }}
th {{ color:#9aa5b1; font-weight:600; }}
tr:hover {{ background:#111826; }}
</style></head>
<body>
<h1>NSE Daily Scanner</h1>
<div class="sub">Last run: {run_date} · Trend-pullback strategy · Not financial advice</div>
<table>
<tr><th>Ticker</th><th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th><th>Shares</th><th>ADX</th><th>RelVol</th></tr>
{rows}
</table>
</body></html>"""
    return html

def main():
    hits = []
    for t in UNIVERSE:
        try:
            r = scan_stock(t)
            if r:
                hits.append(r)
        except Exception:
            pass
    hits.sort(key=lambda x: x["score"], reverse=True)
    top = hits[:5]

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    with open("results.json", "w") as f:
        json.dump({"run_date": run_date, "results": top}, f, indent=2)

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(build_html(top, run_date))

    print(json.dumps(top, indent=2) if top else "No qualifying setups today.")

if __name__ == "__main__":
    main()
