import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from typing import Tuple
from .price_fetcher import PriceFetcher # Relative import

class TrendAnalyzer:
    def __init__(self, price_fetcher: PriceFetcher):
        self.pf = price_fetcher

    def analyze(self, coin: str, days: int, forecast_days: int, vs_currency: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        try:
            chart = self.pf.market_chart(coin, days=days)
            if not chart or "prices" not in chart:
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

            # --- จุดที่แก้ไขบัก: ใช้ pd.to_datetime และ timedelta ให้ถูกต้อง ---
            last_date = pd.to_datetime(hist_df['date'].iloc[-1])
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            
            forecast_df = pd.DataFrame({'date': future_dates, 'price': forecast_prices})

            last_actual_price = hist_df['price'].iloc[-1]
            last_forecast_price = forecast_df['price'].iloc[-1]
            pct_change = ((last_forecast_price - last_actual_price) / last_actual_price) * 100 if last_actual_price != 0 else 0.0

            return hist_df, forecast_df, pct_change

        except Exception as e:
            print(f"Error in TrendAnalyzer for {coin}: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0