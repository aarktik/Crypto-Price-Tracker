from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Alert:
    coin: str
    severity: str
    message: str

@dataclass
class AlertEngine:
    drop_threshold_pct: float = 10.0
    
    def check(self, latest: Dict[str, float], previous: Dict[str, float]) -> List[Alert]:
        alerts: List[Alert] = []
        for coin, cur_price in latest.items():
            prev_price = previous.get(coin)
            if prev_price is None or prev_price == 0:
                continue
            change_pct = ((cur_price - prev_price) / prev_price) * 100.0
            if change_pct <= -abs(self.drop_threshold_pct):
                alerts.append(Alert(
                    coin=coin,
                    severity="warning",
                    message=f"{coin}: Price dropped {abs(change_pct):.2f}% since last snapshot."
                ))
        return alerts
