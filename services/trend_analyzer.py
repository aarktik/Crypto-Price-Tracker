from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class TrendAnalyzer:
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def analyze(self, coin_id: str, days: int = 30, forecast_days: int = 7, vs_currency: str = "usd"):
        data = self.fetcher.market_chart(coin_id, days)
        df = pd.DataFrame(data["prices"], columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.groupby(df["date"].dt.date)["price"].mean().reset_index()
        df["t"] = np.arange(len(df))
        X, y = df[["t"]], df["price"]
        model = LinearRegression().fit(X, y)
        future_t = np.arange(len(df), len(df) + forecast_days)
        forecast = model.predict(future_t.reshape(-1, 1))
        df_forecast = pd.DataFrame({
            "date": pd.to_datetime(df["date"].iloc[-1]) + pd.to_timedelta(np.arange(1, forecast_days+1), unit="D"),
            "price": forecast
        })
        pct_change = ((forecast[-1] - y.iloc[-1]) / y.iloc[-1]) * 100
        return df, df_forecast, pct_change
