import requests
import json

with open('params.json', 'r') as config_file:
    params = json.load(config_file)

url = "https://api.binance.com/api/v3/klines"

response = requests.get(url, params=params)
data = response.json()

ohlcv_data = [{
    "timestamp": entry[0],
    "open": entry[1],
    "high": entry[2],
    "low": entry[3],
    "close": entry[4],
    "volume": entry[5]
} for entry in data]

for ohlcv in ohlcv_data[:5]:
    print(ohlcv)