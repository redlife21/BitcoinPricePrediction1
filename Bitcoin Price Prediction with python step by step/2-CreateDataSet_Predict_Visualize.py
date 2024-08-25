import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#PREPARE (SCALER, SPLİT)
df = pd.read_csv('data/bitcoin_ohlcv.csv')
data = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#ML MODEL
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(20, return_sequences=False))
model.add(Dense(5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=5)

#PREDICTION
future_steps = 60 
future_predictions = []

last_data = test_data[-time_step:].reshape(1, time_step, 1)

for _ in range(future_steps):
    next_pred = model.predict(last_data)
    future_predictions.append(next_pred[0, 0])
    next_pred = next_pred.reshape(1, 1, 1)
    last_data = np.concatenate((last_data[:, 1:, :], next_pred), axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))



#GRAPH
plt.figure(figsize=(14,8))
plt.plot(range(len(scaled_data)), scaler.inverse_transform(scaled_data), label='Mevcut Veri')
future_index = range(len(scaled_data), len(scaled_data) + future_steps)
plt.plot(future_index, future_predictions, label='Gelecek Tahminleri', linestyle='--')
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.show()