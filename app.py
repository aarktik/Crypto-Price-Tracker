from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Dict, List
from services.trend_analyzer import TrendAnalyzer

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from services.price_fetcher import PriceFetcher
from services.data_logger import DataLogger
from services.alert_engine import AlertEngine
from services.summarizer import AISummarizer

st.set_page_config(page_title="Crypto Price Tracker", page_icon="💹", layout="wide")

POPULAR_COINS = [
    "bitcoin","ethereum","binancecoin","solana","ripple","cardano","dogecoin","tron",
    "polygon","litecoin","bitcoin-cash","chainlink","polkadot","avalanche-2",
    "toncoin","shiba-inu","stellar","near","aptos","arbitrum","optimism","uniswap",
    "internet-computer","monero","filecoin","hedera","maker","algorand"
]

def human(n: float) -> str:
    try:
        if n >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
        if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if n >= 1_000: return f"{n/1_000:.2f}K"
        return f"{n:,.2f}"
    except Exception:
        return str(n)

st.title("Crypto Price Tracker — Data Analysis Only")
st.caption("⚠️ For educational/data analysis purposes only. No trading features.")

# Sidebar
st.sidebar.header("Settings")
with st.sidebar.expander("Coins", expanded=True):
    preset = st.radio("Presets", ["Top 5", "Top 10", "Custom"], index=1)
    default = POPULAR_COINS[:5] if preset == "Top 5" else POPULAR_COINS[:10]
    coins = st.multiselect("Select coins (CoinGecko IDs):", options=POPULAR_COINS,
                           default=default if preset != "Custom" else ["bitcoin","ethereum"])

vs_currency = st.sidebar.selectbox("Currency", options=["usd", "thb"], index=0)
days_history = st.sidebar.slider("History (days)", min_value=7, max_value=120, value=30, step=1)
drop_threshold = st.sidebar.slider("Alert: drop ≥ (%)", min_value=1, max_value=50, value=10)

with st.sidebar.expander("API Keys"):
    cg_env = os.getenv("COINGECKO_API_KEY") or ""
    cg_input = st.text_input("COINGECKO_API_KEY", value=cg_env, type="password")
    if cg_input and cg_input != os.getenv("COINGECKO_API_KEY"):
        os.environ["COINGECKO_API_KEY"] = cg_input
        st.success("CoinGecko key set for session.")

    openai_env = os.getenv("OPENAI_API_KEY") or ""
    openai_input = st.text_input("OPENAI_API_KEY (optional for AI summary)", value=openai_env, type="password")
    if openai_input and openai_input != os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_input
        st.success("OpenAI key set for session.")

# Services
pf = PriceFetcher(vs_currency=vs_currency)
dl = DataLogger(data_dir="data")
ae = AlertEngine(drop_threshold_pct=drop_threshold)

# Action Row
left, right = st.columns([1,2], vertical_alignment="center")
with left:
    fetch = st.button("📥 Fetch & Save Daily Snapshot", help="Fetch current prices and upsert daily CSV snapshots")
with right:
    st.info("Snapshots are saved to `data/{coin}_daily.csv` (upsert by date).")

if fetch and coins:
    try:
        latest = pf.current_prices(coins)
        dl.write_snapshot(latest, ts=datetime.now(timezone.utc))
        st.success("Snapshot saved!")
    except Exception as e:
        st.error(f"Fetch failed: {e}")

st.divider()

tab_overview, tab_history = st.tabs(["Overview", "History & 7D MA"])

with tab_overview:
    st.subheader("Market Overview")
    try:
        rows = pf.markets(coins, price_change_percentage="24h,7d")
    except Exception as e:
        rows = []
        st.error(f"Failed to load market data: {e}")

    if rows:
        df = pd.DataFrame([{
            "coin": r.get("id"),
            "symbol": r.get("symbol"),
            "name": r.get("name"),
            f"price ({vs_currency})": r.get("current_price"),
            "24h %": r.get("price_change_percentage_24h_in_currency"),
            "7d %": (r.get("price_change_percentage_7d_in_currency") or r.get("price_change_percentage_7d")),
            "mkt cap": r.get("market_cap")
        } for r in rows])
        if not df.empty:
            df = df.sort_values(by=f"price ({vs_currency})", ascending=False)
            st.dataframe(
                df.style.format({
                    f"price ({vs_currency})": "{:,.2f}",
                    "24h %": "{:+.2f}%",
                    "7d %": "{:+.2f}%",
                    "mkt cap": lambda x: human(x) if pd.notna(x) else ""
                }),
                use_container_width=True,
                hide_index=True
            )

    st.subheader("Alerts vs Last Saved Snapshot")
    try:
        latest_now = pf.current_prices(coins)
    except Exception as e:
        latest_now = {}
        st.error(f"Failed to fetch current prices: {e}")

    previous: Dict[str, float] = {}
    for c in coins:
        hist = dl.load_history(c)
        if not hist.empty:
            previous[c] = float(hist.iloc[-1]["price"])

    alerts = ae.check(latest_now, previous)
    alerts_text = "; ".join([a.message for a in alerts]) if alerts else ""

    if alerts:
        for a in alerts:
            st.warning(f"**{a.coin}** — {a.message}")
    else:
        st.success("No alerts based on last saved snapshot.")

    # AI Summary
    st.subheader("AI Market Summary")
    with st.expander("AI Settings", expanded=False):
        use_ai = st.checkbox("Enable AI Summary", value=True)
        openai_model = st.text_input("OpenAI Model", value=os.getenv("OPENAI_MODEL") or "gpt-4o-mini")

    if use_ai:
        summarizer = AISummarizer(api_key=os.getenv("OPENAI_API_KEY"), model=openai_model)
        summary = summarizer.summarize(rows or [], alerts_text=alerts_text)
        st.info(summary)
    else:
        st.caption("AI summary disabled.")

with tab_history:
    st.subheader("Per-Coin History & 7-Day Moving Average")
    for coin in coins:
        st.markdown(f"### {coin.capitalize()} ({vs_currency.upper()})")
        try:
            chart = pf.market_chart(coin, days=days_history)
            mdf = pd.DataFrame(chart.get("prices", []), columns=["ms", "price"])
            mdf["date"] = pd.to_datetime(mdf["ms"], unit="ms").dt.date
            mdf = mdf.groupby("date", as_index=False)["price"].mean()
        except Exception:
            mdf = pd.DataFrame(columns=["date","price"])

        snap = dl.load_history(coin)
        snap["date"] = snap["date"].dt.date
        if not mdf.empty:
            merged = pd.merge(mdf, snap, on="date", how="outer", suffixes=("_api","_snap"))
            merged["price"] = merged["price_snap"].combine_first(merged["price_api"])
            mdf = merged[["date","price"]].dropna()
        else:
            mdf = snap.rename(columns={"date":"date","price":"price"})

        if mdf.empty:
            st.info("No data yet. Fetch once to create snapshots.")
            continue

        mdf = mdf.sort_values("date")
        mdf["ma7"] = mdf["price"].rolling(window=7).mean()

        fig, ax = plt.subplots()
        ax.plot(mdf["date"], mdf["price"], label="Daily Close")
        ax.plot(mdf["date"], mdf["ma7"], label="7-day MA")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Price ({vs_currency.upper()})")
        ax.set_title(f"{coin.capitalize()} Price & 7-Day MA")
        ax.legend()
        st.pyplot(fig)

tab_trend = st.tabs(["AI Trend Analyzer"])[0]
with tab_trend:
    st.subheader("AI Trend Analyzer (30-Day History + 7-Day Forecast)")
    ta = TrendAnalyzer(pf)
    for coin in coins:
        st.markdown(f"### {coin.capitalize()}")
        try:
            hist, forecast, pct = ta.analyze(coin, days=30, forecast_days=7, vs_currency=vs_currency)
            fig, ax = plt.subplots()
            ax.plot(hist["date"], hist["price"], label="Actual (30 Days)")
            ax.plot(forecast["date"], forecast["price"], "--", label="Forecast (+7 Days)")
            ax.axvline(hist["date"].iloc[-1], color="gray", linestyle=":", label="⚡ Forecast Start")
            ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel(vs_currency.upper())
            st.pyplot(fig)
            trend = "rising" if pct > 0 else "falling"
            st.info(f"**{coin.capitalize()}** appears to be *{trend}* by ~{abs(pct):.2f}% over the next 7 days.")
        except Exception as e:
            st.error(f"{coin}: {e}")


st.caption("© 2025 — Educational project. Data source: CoinGecko API")
