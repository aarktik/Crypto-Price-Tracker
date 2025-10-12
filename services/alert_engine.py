from typing import Dict, List

class Alert:
    def __init__(self, coin: str, message: str):
        self.coin = coin
        self.message = message

class AlertEngine:
    def __init__(self, drop_threshold_pct: float):
        self.drop_threshold_pct = drop_threshold_pct

    def check(self, current: Dict[str, float], previous: Dict[str, float]) -> List[Alert]:
        alerts = []
        for coin, current_price in current.items():
            if coin in previous and previous[coin] > 0:
                prev_price = previous[coin]
                drop_pct = ((prev_price - current_price) / prev_price) * 100
                if drop_pct >= self.drop_threshold_pct:
                    alerts.append(Alert(coin, f'Dropped by {drop_pct:.2f}% since last snapshot.'))
        return alerts