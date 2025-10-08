from __future__ import annotations
import os
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_KEY = "sk-API_KEY"


SYSTEM_PROMPT = (
    "You are a concise crypto market analyst. "
    "Summarize key movements, highlight top gainers/losers, and mention any alerts. "
    "Keep it factual, neutral, and under 120 words. Avoid financial advice."
)

@dataclass
class AISummarizer:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = 0.3

    def summarize(self, rows: List[Dict], alerts_text: str = "") -> str:
        if not (self.api_key or os.getenv("OPENAI_API_KEY")):
            return self._fallback_summary(rows, alerts_text)

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        try:
            user_text = self._build_user_prompt(rows, alerts_text)
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
            }
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            r = requests.post(f"{OPENAI_BASE}/chat/completions", json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_summary(rows, alerts_text)

    def _build_user_prompt(self, rows: List[Dict], alerts_text: str) -> str:
        lines = []
        for r in rows[:20]:
            name = r.get("name") or r.get("id")
            price = r.get("current_price")
            d1 = r.get("price_change_percentage_24h_in_currency")
            d7 = r.get("price_change_percentage_7d_in_currency") or r.get("price_change_percentage_7d")
            lines.append(f"{name}: price={price}, 24h%={d1}, 7d%={d7}")
        body = "\n".join(lines) or "No market data."
        if alerts_text:
            body += f"\nAlerts: {alerts_text}"
        return body

    def _fallback_summary(self, rows: List[Dict], alerts_text: str) -> str:
        if not rows:
            base = "No market data available yet."
        else:
            sorted_24h = sorted(rows, key=lambda r: (r.get('price_change_percentage_24h_in_currency') or 0), reverse=True)
            top = sorted_24h[0]
            worst = sorted_24h[-1]
            base = (
                f"Top 24h: {top.get('name')} ({top.get('symbol')}) "
                f"{(top.get('price_change_percentage_24h_in_currency') or 0):+.2f}%. "
                f"Worst 24h: {worst.get('name')} ({worst.get('symbol')}) "
                f"{(worst.get('price_change_percentage_24h_in_currency') or 0):+.2f}%."
            )
        if alerts_text:
            base += f" Alerts: {alerts_text}"
        return base
