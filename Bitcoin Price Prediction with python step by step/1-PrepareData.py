import pandas as pd
import requests
import os
import json

with open('params.json', 'r') as config_file:
    params = json.load(config_file)

url = "https://api.binance.com/api/v3/klines"

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])

df = df[["timestamp", "open", "high", "low", "close", "volume"]]

os.makedirs('data', exist_ok=True)

csv_file_path = os.path.join('data', 'bitcoin_ohlcv.csv')

df.to_csv(csv_file_path, index=False)

print(f"CSV dosyası '{csv_file_path}' konumuna kaydedildi.")