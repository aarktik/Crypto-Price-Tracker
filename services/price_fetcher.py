from __future__ import annotations
import os
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def _headers() -> dict:
    key = os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
    headers = {"accept": "application/json"}
    if key:
        headers["x-cg-demo-api-key"] = key
    return headers

@dataclass
class PriceFetcher:
    vs_currency: str = "usd"
    session: Optional[requests.Session] = None

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict | List[Dict] | Any:
        s = self.session or requests.Session()
        url = f"{COINGECKO_BASE}{path}"
        r = s.get(url, params=params, headers=_headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def current_prices(self, coin_ids: List[str]) -> Dict[str, float]:
        if not coin_ids:
            return {}
        params = {"ids": ",".join(coin_ids), "vs_currencies": self.vs_currency}
        data = self._get("/simple/price", params=params)
        return {c: float(data.get(c, {}).get(self.vs_currency, float("nan"))) for c in coin_ids}

    def market_chart(self, coin_id: str, days: int = 30) -> Dict:
        params = {"vs_currency": self.vs_currency, "days": days}
        return self._get(f"/coins/{coin_id}/market_chart", params=params)

    def markets(self, coin_ids: List[str], price_change_percentage: str = "24h,7d") -> List[Dict]:
        if not coin_ids:
            return []
        params = {
            "vs_currency": self.vs_currency,
            "ids": ",".join(coin_ids),
            "price_change_percentage": price_change_percentage,
            "per_page": len(coin_ids),
            "page": 1
        }
        data = self._get("/coins/markets", params=params)
        return data if isinstance(data, list) else []
