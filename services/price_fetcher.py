import os
import requests
import pandas as pd
from typing import List, Dict

class PriceFetcher:
    def __init__(self, vs_currency: str = "usd"):
        self.vs_currency = vs_currency
        self.api_base = "https://api.coingecko.com/api/v3"
        self.api_key = os.getenv("COINGECKO_API_KEY")

    def _make_request(self, endpoint: str, params: dict = None):
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        try:
            response = requests.get(f"{self.api_base}/{endpoint}", params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return None

    def current_prices(self, coins: List[str]) -> Dict[str, float]:
        params = {
            "ids": ",".join(coins),
            "vs_currencies": self.vs_currency,
            "include_market_cap": "false",
            "include_24hr_vol": "false",
            "include_24hr_change": "false",
            "include_last_updated_at": "false"
        }
        data = self._make_request("simple/price", params)
        if data:
            return {coin: details[self.vs_currency] for coin, details in data.items()}
        return {}

    def markets(self, coins: List[str], price_change_percentage: str = "24h,7d") -> List[Dict]:
        params = {
            "vs_currency": self.vs_currency,
            "ids": ",".join(coins),
            "order": "market_cap_desc",
            "per_page": len(coins),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": price_change_percentage
        }
        return self._make_request("coins/markets", params) or []

    def market_chart(self, coin: str, days: int) -> Dict:
        params = {
            "vs_currency": self.vs_currency,
            "days": days,
            "interval": "daily"
        }
        return self._make_request(f"coins/{coin}/market_chart", params) or {}