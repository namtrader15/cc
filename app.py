from flask import Flask, jsonify
from binance.client import Client
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime
import time

app = Flask(__name__)

# Đoạn mã của anh được tích hợp trong các hàm dưới đây
def calculate_combined_probability(p1=None, p2=None, p3=None):
    p1 = p1 if p1 is not None else 0
    p2 = p2 if p2 is not None else 0
    p3 = p3 if p3 is not None else 0
    P_A = p1 + p2 + p3 - (p1 * p2) - (p1 * p3) - (p2 * p3) + (p1 * p2 * p3)
    return P_A

def get_realtime_klines(symbol, interval, lookback, client, end_time=None):
    if end_time:
        klines = client.futures_klines(symbol=symbol, interval=interval, endTime=int(end_time.timestamp() * 1000), limit=lookback)
    else:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    data['volume'] = data['volume'].astype(float)
    
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
    ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)
    
    data['open'] = ha_open
    data['high'] = ha_high
    data['low'] = ha_low
    data['close'] = ha_close
    
    return data

def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def analyze_trend(interval, client):
    symbol = 'BTCUSDT'
    lookback = 1000
    data = get_realtime_klines(symbol, interval, lookback, client)
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    features = data[['rsi', 'macd', 'signal_line']].dropna()
    target = data['target'].dropna()
    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs']}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, refit=True, verbose=0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = grid.predict_proba(latest_features)[0]
    prediction = grid.predict(latest_features)
    threshold = 0.45
    if prediction_prob[1] > 1 - threshold:
        trend = "Tăng"
    elif prediction_prob[1] < threshold:
        trend = "Giảm"
    else:
        trend = "Xu hướng không rõ ràng"
    current_price = data['close'].iloc[-1]
    return {"trend": trend, "accuracy": accuracy, "current_price": current_price}

# Route để tích hợp mã Python của anh
@app.route('/analyze')
def analyze():
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'
    client = Client(api_key, api_secret, tld='com', testnet=False)
    
    # Thực hiện phân tích cho khung thời gian 1 phút
    result = analyze_trend(Client.KLINE_INTERVAL_1MINUTE, client)
    
    return jsonify(result)  # Trả về kết quả dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True)
