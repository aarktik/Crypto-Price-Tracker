# 💹 Crypto Price Tracker v4 — AI Trend Analyzer

A Streamlit web app that analyzes crypto prices, shows 7-day forecasts using AI, 
and provides automatic summaries powered by OpenAI GPT-4o-mini.

## Features
- 📊 30-day historical prices with linear regression forecast (+7 days)
- 🤖 AI market summary (English)
- ⚡ Real-time alert engine
- 🎨 Dark/Blue modern theme
- 🧩 OOP modular structure (PriceFetcher, DataLogger, AlertEngine, AISummarizer, TrendAnalyzer)

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
