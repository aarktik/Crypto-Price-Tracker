# =============================================================================
# ⚠️  WARNING: API KEYS ARE HARDCODED FOR PERSONAL USE ONLY ⚠️
# Embedding API keys directly in the source code is a security risk.
# This is NOT recommended for production or public projects.
# For better security, use environment variables.
# =============================================================================

from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression

# --- ฝัง API Key ที่นี่ ---
COINGECKO_API_KEY = "CG-sLEJuPxmZkmLWWs2wnQ7iYLA"
OPENAI_API_KEY = "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"
USD_TO_THB_RATE = 36.5

# --- เริ่มต้น: ส่วนของ Service ที่ถูกฝังเข้ามา ---

class PriceFetcher:
    """ดึงข้อมูลราคาจาก CoinGecko API (ใช้ข้อมูลจริง และมีข้อมูลจำลองเป็นตัวเลือกสำรอง)"""
    def __init__(self, vs_currency: str = "usd"):
        self.vs_currency = vs_currency
        self.api_base = "https://api.coingecko.com/api/v3"
        self.api_key = COINGECKO_API_KEY

    def _get_mock_data(self, coins: List[str]) -> List[Dict]:
        """สร้างข้อมูลจำลองที่ใกล้เคียงความเป็นจริง และรองรับการแปลงสกุลเงิน"""
        if not coins: return [] # Return empty list if no coins are selected
        mock_data = []
        base_prices_usd = {"bitcoin": 67000, "ethereum": 3500, "binancecoin": 610, "solana": 180, "ripple": 0.52}
        for coin in coins:
            base_price_usd = base_prices_usd.get(coin, np.random.uniform(0.01, 1000))
            price_usd = base_price_usd * (1 + np.random.uniform(-0.05, 0.05))
            if self.vs_currency == "thb":
                price = price_usd * USD_TO_THB_RATE
                market_cap = price * np.random.uniform(1_000_000, 25_000_000)
                volume = price * np.random.uniform(50_000, 800_000)
            else:
                price = price_usd
                market_cap = price * np.random.uniform(1_000_000, 25_000_000)
                volume = price * np.random.uniform(50_000, 800_000)
            mock_data.append({
                "id": coin, "symbol": coin[:3].upper(), "name": coin.capitalize(),
                "current_price": price,
                "price_change_percentage_24h_in_currency": np.random.uniform(-8, 8),
                "price_change_percentage_7d_in_currency": np.random.uniform(-15, 15),
                "market_cap": market_cap,
                "total_volume": volume
            })
        return mock_data

    def _make_request(self, endpoint: str, params: dict = None):
        headers = {"x-cg-demo-api-key": self.api_key}
        try:
            response = requests.get(f"{self.api_base}/{endpoint}", params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}. Falling back to mock data.")
            return None

    def current_prices(self, coins: List[str]) -> Dict[str, float]:
        if not coins: return {}
        params = {"ids": ",".join(coins), "vs_currencies": self.vs_currency, "include_market_cap": "false", "include_24hr_vol": "false", "include_24hr_change": "false", "include_last_updated_at": "false"}
        data = self._make_request("simple/price", params)
        if data:
            return {coin: details[self.vs_currency] for coin, details in data.items()}
        mock_data = self._get_mock_data(coins)
        return {item["id"]: item["current_price"] for item in mock_data}

    def markets(self, coins: List[str], price_change_percentage: str = "24h,7d") -> List[Dict]:
        if not coins: return []
        params = {"vs_currency": self.vs_currency, "ids": ",".join(coins), "order": "market_cap_desc", "per_page": len(coins), "page": 1, "sparkline": "false", "price_change_percentage": price_change_percentage}
        data = self._make_request("coins/markets", params)
        if data:
            return data
        return self._get_mock_data(coins)

    def market_chart(self, coin: str, days: int) -> Dict:
        params = {"vs_currency": self.vs_currency, "days": days, "interval": "daily"}
        data = self._make_request(f"coins/{coin}/market_chart", params)
        if data:
            return data
        dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=days, freq='D'))
        base_price = np.random.uniform(100, 70000)
        if self.vs_currency == "thb": base_price *= USD_TO_THB_RATE
        prices = base_price + np.cumsum(np.random.randn(days) * base_price * 0.02)
        return {"prices": [[int(d.timestamp() * 1000), p] for d, p in zip(dates, prices)]}

class DataLogger:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def write_snapshot(self, data: Dict[str, float], ts: datetime):
        for coin, price in data.items():
            file_path = os.path.join(self.data_dir, f"{coin}_daily.csv")
            new_row = pd.DataFrame({"date": [ts.date()], "price": [price]})
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=["date"])
                df = pd.concat([df, new_row]).drop_duplicates(subset=["date"], keep="last")
            else:
                df = new_row
            df.to_csv(file_path, index=False)

    def load_history(self, coin: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{coin}_daily.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "price"])

class AlertEngine:
    def __init__(self, drop_threshold_pct: float):
        self.drop_threshold_pct = drop_threshold_pct

    def check(self, current: Dict[str, float], previous: Dict[str, float]) -> List[object]:
        alerts = []
        for coin, current_price in current.items():
            if coin in previous and previous[coin] > 0:
                prev_price = previous[coin]
                drop_pct = ((prev_price - current_price) / prev_price) * 100
                if drop_pct >= self.drop_threshold_pct:
                    alert = type('Alert', (), {'coin': coin, 'message': f'Dropped by {drop_pct:.2f}% since last snapshot.'})()
                    alerts.append(alert)
        return alerts

class AISummarizer:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model

    def summarize(self, market_data: List[Dict], alerts_text: str) -> str:
        if not self.api_key:
            return "AI Summary disabled: OpenAI API key not provided."
        return f"🤖 AI Summary: The market is showing mixed signals. {alerts_text} Key movements include Bitcoin and Ethereum. (This is a placeholder summary)."

class TrendAnalyzer:
    def __init__(self, price_fetcher: PriceFetcher):
        self.pf = price_fetcher

    def analyze(self, coin: str, days: int, forecast_days: int, vs_currency: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        try:
            chart = self.pf.market_chart(coin, days=days)
            if not chart or "prices" not in chart or not chart["prices"]:
                return pd.DataFrame(), pd.DataFrame(), 0.0

            hist_df = pd.DataFrame(chart.get("prices", []), columns=["ms", "price"])
            hist_df['date'] = pd.to_datetime(hist_df['ms'], unit='ms')
            hist_df = hist_df.groupby('date', as_index=False)['price'].mean()
            hist_df = hist_df.sort_values('date')

            if len(hist_df) < 2:
                return hist_df, pd.DataFrame(), 0.0

            hist_df['day_number'] = np.arange(len(hist_df))
            X = hist_df[['day_number']]
            y = hist_df['price']
            model = LinearRegression()
            model.fit(X, y)

            last_day_number = hist_df['day_number'].iloc[-1]
            future_day_numbers = np.arange(last_day_number + 1, last_day_number + forecast_days + 1).reshape(-1, 1)
            forecast_prices = model.predict(future_day_numbers)

            last_date = pd.to_datetime(hist_df['date'].iloc[-1])
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            
            forecast_df = pd.DataFrame({'date': future_dates, 'price': forecast_prices})

            last_actual_price = hist_df['price'].iloc[-1]
            last_forecast_price = forecast_df['price'].iloc[-1]
            pct_change = ((last_forecast_price - last_actual_price) / last_actual_price) * 100 if last_actual_price != 0 else 0.0

            return hist_df, forecast_df, pct_change
        except Exception as e:
            st.error(f"Error in TrendAnalyzer for {coin}: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0

# --- ส่วนใหม่: Discord Notifier Service ---
class DiscordNotifier:
    """ส่งการแจ้งเตือนไปยัง Discord ผ่าน Webhook"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, alert: object, current_price: float, currency: str):
        if not self.webhook_url:
            return

        # สร้างข้อความในรูปแบบ Embed ของ Discord
        embed = {
            "title": f"🚨 Price Drop Alert: {alert.coin.upper()}",
            "description": alert.message,
            "color": 0xE94560,  # สีแดง
            "fields": [
                {
                    "name": "Current Price",
                    "value": f"{'฿' if currency == 'thb' else '$'}{current_price:,.4f}",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Crypto Price Tracker"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        data = {"embeds": [embed]}

        try:
            response = requests.post(self.webhook_url, json=data)
            response.raise_for_status()  # จะเกิด Error ถ้าส่งไม่สำเร็จ (status code ไม่ใช่ 2xx)
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to send Discord notification: {e}")
            return False

# --- สิ้นสุดส่วนของ Service ---


st.set_page_config(page_title="Crypto Price Tracker", page_icon="💹", layout="wide", initial_sidebar_state="expanded")

# --- กำหนดธีมและสไตล์ UI ---
PRIMARY_COLOR = "#0F3460"
SECONDARY_COLOR = "#E94560"
SUCCESS_COLOR = "#53D769"
WARNING_COLOR = "#F2B90F"
BACKGROUND_COLOR = "#F5F7FA"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR}; }}
    .main-header {{ background: linear-gradient(to right, {PRIMARY_COLOR}, {SECONDARY_COLOR}); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    .main-header h1 {{ margin: 0; font-size: 2.5rem; font-weight: 700; }}
    .main-header p {{ margin: 0.5rem 0 0 0; opacity: 0.9; }}
    .card {{ background-color: {CARD_COLOR}; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 1.5rem; border: 1px solid #e0e0e0; }}
    h1, h2, h3 {{ color: {PRIMARY_COLOR}; }}
    .stTabs [data-baseweb="tab-list"] {{ background-color: {CARD_COLOR}; border-radius: 10px; padding: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 600; color: {TEXT_COLOR}; }}
    .stTabs [aria-selected="true"] {{ background-color: {PRIMARY_COLOR}; color: white; }}
    div.stButton > button:first-child {{ background-color: {PRIMARY_COLOR}; color: white; font-weight: bold; border-radius: 8px; padding: 0.75rem 1.5rem; transition: all 0.3s ease; border: none; width: 100%; }}
    div.stButton > button:first-child:hover {{ background-color: {SECONDARY_COLOR}; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }}
    .dataframe {{ border-radius: 10px; overflow: hidden; }}
    .alert-success {{ background-color: rgba(83, 215, 105, 0.1); padding: 1rem; border-radius: 8px; border-left: 5px solid {SUCCESS_COLOR}; }}
    .alert-warning {{ background-color: rgba(242, 185, 15, 0.1); padding: 1rem; border-radius: 8px; border-left: 5px solid {WARNING_COLOR}; }}
    .alert-danger {{ background-color: rgba(233, 69, 96, 0.1); padding: 1rem; border-radius: 8px; border-left: 5px solid {SECONDARY_COLOR}; }}
    .alert-info {{ background-color: rgba(15, 52, 96, 0.1); padding: 1rem; border-radius: 8px; border-left: 5px solid {PRIMARY_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# --- ฟังก์ชันช่วยเหลือ ---
POPULAR_COINS = ["bitcoin","ethereum","binancecoin","solana","ripple","cardano","dogecoin","tron","polygon","litecoin"]

def human(n: float, currency: str) -> str:
    symbol = "฿" if currency == "thb" else "$"
    if n >= 1_000_000_000: return f"{symbol}{n/1_000_000_000:.2f}B"
    if n >= 1_000_000: return f"{symbol}{n/1_000_000:.2f}M"
    if n >= 1_000: return f"{symbol}{n/1_000:.2f}K"
    return f"{symbol}{n:,.2f}"

def format_change(val):
    if val is None: return "N/A"
    color = SUCCESS_COLOR if val >= 0 else SECONDARY_COLOR
    arrow = "📈" if val >= 0 else "📉"
    return f'<span style="color:{color};font-weight:bold">{arrow} {abs(val):.2f}%</span>'

# --- ส่วนหัวของแอปพลิเคชัน ---
st.markdown("""
<div class="main-header">
    <h1>💹 Crypto Price Tracker</h1>
    <p>Real-time Cryptocurrency Analysis & Tracking Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    with st.expander("🪙 Select Coins", expanded=True):
        coins = st.multiselect("Choose cryptocurrencies to track:", options=POPULAR_COINS, default=["bitcoin", "ethereum", "binancecoin"])
    
    vs_currency = st.selectbox("💱 Display Currency", options=["usd", "thb"], index=0, format_func=lambda x: "USD" if x == "usd" else "THB")
    days_history = st.slider("📊 History (days)", min_value=7, max_value=120, value=30, step=1)
    drop_threshold = st.slider("🚨 Alert Drop Threshold (%)", min_value=1, max_value=50, value=10)
    
    st.markdown("---")
    st.markdown("### 🔔 Notifications")
    # --- ส่วนใหม่: ช่องใส่ Discord Webhook ---
    discord_webhook = st.text_input(
        "Discord Webhook URL", 
        type="password", 
        help="Enter your Discord channel webhook URL to receive price drop alerts."
    )
    st.caption("Don't have one? Create one in your Discord channel settings > Integrations.")

    st.markdown("---")
    st.info("API Keys are configured in this version.", icon="🔑")

# --- Services ---
pf = PriceFetcher(vs_currency=vs_currency)
dl = DataLogger(data_dir="data")
ae = AlertEngine(drop_threshold_pct=drop_threshold)
ta = TrendAnalyzer(pf)
summarizer = AISummarizer(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
# --- ส่วนใหม่: สร้าง instance ของ DiscordNotifier ---
discord_notifier = DiscordNotifier(webhook_url=discord_webhook)

# --- Action Row ---
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    fetch = st.button("🔄 Fetch Latest Data & Save Snapshot", use_container_width=True)
with col2:
    if st.button("📊 Refresh View", use_container_width=True):
        st.rerun()
with col3:
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")
st.markdown('</div>', unsafe_allow_html=True)

if fetch and coins:
    with st.spinner("Fetching data..."):
        try:
            latest = pf.current_prices(coins)
            dl.write_snapshot(latest, ts=datetime.now(timezone.utc))
            st.markdown('<div class="alert-success">✅ Data fetched and saved successfully!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="alert-danger">❌ Fetch failed: {e}</div>', unsafe_allow_html=True)

# --- Main Content with Tabs ---
tab_overview, tab_history, tab_trend = st.tabs(["📊 Market Overview", "📈 Historical Analysis", "🤖 AI Forecast"])

with tab_overview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Market at a Glance")
    
    rows = []
    alerts = []

    if not coins:
        st.warning("⚠️ Please select at least one coin from the sidebar to see the market overview.")
    else:
        rows = pf.markets(coins, price_change_percentage="24h,7d")
        if rows:
            df = pd.DataFrame([{
                "name": r.get("name"), "symbol": r.get("symbol").upper(),
                "price": r.get("current_price"), "24h_change": r.get("price_change_percentage_24h_in_currency"),
                "7d_change": r.get("price_change_percentage_7d_in_currency"), "market_cap": r.get("market_cap")
            } for r in rows])
            df = df.sort_values(by="market_cap", ascending=False)
            
            total_mcap = df['market_cap'].sum()
            avg_change = df['24h_change'].mean()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Market Cap", human(total_mcap, vs_currency))
            col2.metric("Avg. 24h Change", f"{avg_change:.2f}%")
            col3.metric("Coins Tracked", len(df))
            
            styled_df = df.style.format({
                "price": lambda x: human(x, vs_currency), 
                "24h_change": lambda x: format_change(x), 
                "7d_change": lambda x: format_change(x),
                "market_cap": lambda x: human(x, vs_currency) if pd.notna(x) else ""
            }).set_properties(**{
                'text-align': 'center',
                'border': '1px solid #e0e0e0'
            }).set_table_styles([{
                'selector': 'thead th',
                'props': [
                    ('background-color', PRIMARY_COLOR), 
                    ('color', 'white'), 
                    ('font-weight', 'bold'),
                    ('border', '1px solid #e0e0e0')
                ]
            },{
                'selector': 'td',
                'props': [('border', '1px solid #e0e0e0')]
            }])
            
            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        else:
            st.error("❌ Could not retrieve market data. Please check your API key or network connection.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚨 Price Drop Alerts")
    if not coins:
        st.info("Select coins to enable alerts.")
    else:
        latest_now = pf.current_prices(coins)
        previous = {c: dl.load_history(c).iloc[-1]["price"] for c in coins if not dl.load_history(c).empty}
        alerts = ae.check(latest_now, previous)
        if alerts:
            for a in alerts:
                st.markdown(f'<div class="alert-warning">⚠️ <strong>{a.coin.upper()}</strong> — {a.message}</div>', unsafe_allow_html=True)
                # --- ส่วนใหม่: ส่งการแจ้งเตือนไปยัง Discord ---
                if discord_webhook:
                    # ดึงราคาปัจจุบันจากข้อมูลล่าสุด
                    current_price_for_alert = latest_now.get(a.coin, 0)
                    if discord_notifier.send_alert(a, current_price_for_alert, vs_currency):
                        st.success(f"✅ Alert for {a.coin.upper()} sent to Discord!")
        else:
            st.markdown('<div class="alert-success">✅ No significant price drops detected.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Market Summary")
    alerts_text = "; ".join([a.message for a in alerts]) if alerts else "No active alerts."
    summary = summarizer.summarize(rows, alerts_text)
    st.markdown(f'<div class="alert-info">{summary}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


with tab_history:
    if not coins:
        st.warning("⚠️ Please select at least one coin from the sidebar to view its history.")
    else:
        for coin in coins:
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.subheader(f"📈 {coin.upper()} Price History & 7-Day MA")
            try:
                chart = pf.market_chart(coin, days=days_history)
                if not chart or "prices" not in chart or not chart["prices"]:
                    st.error(f"Could not retrieve chart data for {coin}.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                mdf = pd.DataFrame(chart.get("prices", []), columns=["ms", "price"])
                mdf['date'] = pd.to_datetime(mdf['ms'], unit='ms')
                mdf = mdf.groupby('date', as_index=False)['price'].mean()
                mdf = mdf.sort_values("date")
                mdf["ma7"] = mdf["price"].rolling(window=7, min_periods=1).mean()

                fig, ax = plt.subplots(figsize=(12, 6), facecolor=CARD_COLOR)
                ax.plot(mdf["date"], mdf["price"], label="Daily Price", color=PRIMARY_COLOR, linewidth=2.5)
                ax.plot(mdf["date"], mdf["ma7"], label="7-Day Moving Average", color=WARNING_COLOR, linestyle='--', linewidth=2)
                ax.set_facecolor(BACKGROUND_COLOR)
                ax.set_title(f"{coin.capitalize()} Price Trend", fontsize=16, color=PRIMARY_COLOR)
                ax.set_xlabel("Date"); ax.set_ylabel(f"Price ({vs_currency.upper()})")
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate chart for {coin}: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

with tab_trend:
    st.subheader("🤖 AI-Powered 7-Day Price Forecast")
    st.caption("Forecasts are generated using a linear regression model and are for educational purposes only.")
    if not coins:
        st.warning("⚠️ Please select at least one coin from the sidebar to view its forecast.")
    else:
        for coin in coins:
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Forecast for {coin.upper()}")
            try:
                hist, forecast, pct = ta.analyze(coin, days=30, forecast_days=7, vs_currency=vs_currency)
                if hist.empty:
                    st.info("Not enough data to generate a forecast.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                fig, ax = plt.subplots(figsize=(12, 6), facecolor=CARD_COLOR)
                ax.plot(hist["date"], hist["price"], label="Historical Price (30d)", color=PRIMARY_COLOR, marker='o', linewidth=2)
                ax.plot(forecast["date"], forecast["price"], label="Forecast (7d)", color=SECONDARY_COLOR, linestyle='--', marker='x', linewidth=2.5)
                ax.axvline(hist["date"].iloc[-1], color="gray", linestyle=":", label="Forecast Start")
                ax.set_facecolor(BACKGROUND_COLOR)
                ax.set_title(f"{coin.capitalize()} Price Forecast", fontsize=16, color=PRIMARY_COLOR)
                ax.set_xlabel("Date"); ax.set_ylabel(f"Price ({vs_currency.upper()})")
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                st.pyplot(fig, use_container_width=True)
                
                trend = "📈 Rising" if pct > 0 else "📉 Falling"
                st.markdown(f'<div class="alert-info">The model predicts a <strong>{trend}</strong> trend of ~{abs(pct):.2f}% over the next 7 days.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not generate forecast for {coin}: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; margin-top: 2rem; color: #888;">© 2025 Crypto Price Tracker — Educational Project | Data Source: CoinGecko API</div>', unsafe_allow_html=True)