import os
import pandas as pd
from datetime import datetime
from typing import Dict

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