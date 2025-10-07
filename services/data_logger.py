from __future__ import annotations
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Dict

class DataLogger:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _csv_path(self, coin_id: str) -> str:
        return os.path.join(self.data_dir, f"{coin_id}_daily.csv")

    def write_snapshot(self, prices: Dict[str, float], ts: datetime | None = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        date_str = ts.date().isoformat()
        for coin, price in prices.items():
            path = self._csv_path(coin)
            row = {"date": date_str, "price": float(price)}
            if os.path.exists(path):
                df = pd.read_csv(path)
                if (df["date"] == date_str).any():
                    df.loc[df["date"] == date_str, "price"] = float(price)
                else:
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            df.to_csv(path, index=False)

    def load_history(self, coin_id: str) -> pd.DataFrame:
        path = self._csv_path(coin_id)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["date", "price"])
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        return df.sort_values("date")
