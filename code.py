from binance.client import Client
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime
import time
from playsound import playsound

def calculate_combined_probability(p1=None, p2=None, p3=None):
    # Kiểm tra các giá trị đầu vào, nếu None thì thay bằng 0
    p1 = p1 if p1 is not None else 0
    p2 = p2 if p2 is not None else 0
    p3 = p3 if p3 is not None else 0
    
    # Tính toán xác suất kết hợp
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
    
    # Tính giá trị Heikin-Ashi
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
    ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2  # Khởi tạo giá trị HA_open đầu tiên
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)
    
    # Thay thế giá trị nến Nhật bằng giá trị Heikin-Ashi trong DataFrame
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

def analyze_trend(interval, name, client):
    # Lấy dữ liệu thời gian thực
    symbol = 'BTCUSDT'
    lookback = 1000  # Giới hạn số lượng nến tối đa là 1000
    data = get_realtime_klines(symbol, interval, lookback, client)
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)

    # Tạo biến target cho học máy (1: giá tăng, 0: giá giảm)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Chuẩn bị dữ liệu cho mô hình học máy
    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    features = data[['rsi', 'macd', 'signal_line']].dropna()
    target = data['target'].dropna()

    # Đảm bảo rằng features và target có cùng số lượng hàng
    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Tuning Hyperparameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, refit=True, verbose=0)
    grid.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = grid.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Dự đoán xu hướng giá thời gian thực
    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = grid.predict_proba(latest_features)[0]
    prediction = grid.predict(latest_features)

    # Ngưỡng cho xu hướng không rõ ràng
    threshold = 0.45

    # Xác định xu hướng dựa trên ngưỡng
    if prediction_prob[1] > 1 - threshold:
        trend = "Tăng"
    elif prediction_prob[1] < threshold:
        trend = "Giảm"
    else:
        trend = "Xu hướng không rõ ràng"

    # Lấy giá hiện tại và volume
    current_price = data['close'].iloc[-1]
    current_volume = data['volume'].iloc[-1]

    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"{name} - Thời gian: {current_time}")
    print(f"  - Accuracy: {accuracy:.2f}")
    print(f"  - F1 Score: {f1:.2f}")
    print(f"  - Dự báo xu hướng: {trend} ({prediction_prob[1]:.2f})")
    print(f"  - Giá hiện tại: {current_price:.2f} USD")
    print(f"  - Volume hiện tại: {current_volume:.2f}")

    return (prediction[0], accuracy) if trend != "Xu hướng không rõ ràng" else (None, None)

def main(counter):
    # Khởi tạo client Binance với API Key và Secret
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'
    client = Client(api_key, api_secret, tld='com', testnet=False)

    # Phân tích xu hướng cho ba khung thời gian
    trend_1m, accuracy_1m = analyze_trend(Client.KLINE_INTERVAL_1MINUTE, "M1", client)
    trend_15m, accuracy_15m = analyze_trend(Client.KLINE_INTERVAL_15MINUTE, "M15", client)
    trend_4h, accuracy_4h = analyze_trend(Client.KLINE_INTERVAL_4HOUR, "H4", client)

    # Kiểm tra kết quả từ ba khung thời gian
    if trend_1m == 1 and trend_15m == 1 and trend_4h == 1:
        print("Xu Hướng Tăng!")
        playsound(r"C:\Users\DELL\Desktop\GPT train\SOUND.mp3")
    elif trend_1m == 0 and trend_15m == 0 and trend_4h == 0:
        print("Xu Hướng Giảm!")
        playsound(r"C:\Users\DELL\Desktop\GPT train\SOUND.mp3")
    elif trend_1m is None and trend_15m is None and trend_4h is None:
        print("Xu hướng không rõ ràng rồi, đóng lệnh đi ông!")
    else:
        print("Xu hướng chưa rõ ràng!")

    # Tính xác suất kết hợp của cả ba mô hình khi chúng đồng thuận
    if isinstance(accuracy_1m, float) and isinstance(accuracy_15m, float) and isinstance(accuracy_4h, float):
        combined_accuracy = calculate_combined_probability(p1=accuracy_1m, p2=accuracy_15m, p3=accuracy_4h)
        print(f"Xác suất đúng của mô hình là: {combined_accuracy * 100:.2f}%")

    # In ra thông điệp BACKTEST
    print(f"BACKTEST {counter}")

if __name__ == "__main__":
    counter = 1
    while True:
        main(counter)
        counter += 1
        time.sleep(10)  # Chờ 10 giây trước khi chạy lại chương trình
