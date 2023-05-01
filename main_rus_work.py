import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# # Получение данных из файла data.csv
# df = pd.read_csv('data.csv')
# data = df['price'].values
# data['date'] = pd.to_datetime(data['date'])
# df.plot (x='date', y='price')
# plt.show ()

# Получение данных из файла data.csv
df = pd.read_csv('data.csv')
data = df['price'].values
#df['date'] = pd.to_datetime(df['date'])


# Преобразование столбца 'date' в тип datetime
df['date'] = pd.to_datetime(df['date'])
#Выводим график из имеющихся данных
df.plot(x='date', y='price')
plt.show()

# Установка столбца 'date' в качестве индекса
df.set_index('date', inplace=True)

# Вычисление скользящего среднего за последние 3 часа
df['rolling_mean'] = df['price'].rolling('3H').mean()

# Получение последней строки из DataFrame
last_row = df.iloc[-1]

# Получение времени и цены из последней строки
last_time = last_row.name.strftime('%Y-%m-%d %H:%M:%S')
last_price = last_row['price']

# Нормализация данных
data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Разделение данных на обучающую и тестовую выборки
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

# Функция для создания набора данных для временных рядов
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# Создание наборов данных для обучения и тестирования
look_back = 10
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Изменение формы входных данных для соответствия требованиям LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Создание и обучение LSTM-модели
model = Sequential()
model.add(LSTM(8, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=32)
# Сохраняем обученную модель
model.save('model.h5')

# Предсказание курса биткоина на ближайший час
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Вычисляем среднеквадратичную ошибку (MSE) и среднюю абсолютную ошибку (MAE)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')

# Подготовка входных данных для прогнозирования на час вперед
last_data = data[-look_back:]
last_data = last_data.reshape((1, look_back, 1))

def predict_1_hour_ahead(data):
    # Получение прогноза на час вперед
    prediction = model.predict(last_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction

# вызовите эту функцию после анализа данных
prediction = predict_1_hour_ahead(data)

# # Получение прогноза на час вперед
# prediction = model.predict(last_data)
# prediction = scaler.inverse_transform(prediction)

# Прогнозируемые значения цены на биткоин в течение следующего часа сохраняю в файл
predictions = pd.DataFrame({'symbol': 'BTCUSDT', 'price': predicted_prices, 'date': predicted_dates})
predictions.to_csv('predictions.csv', index=False)

print(f'Прогноз на час вперед: {prediction[0][0]:.2f}')


# Получение максимальной и минимальной цены
max_price = np.max(prediction)
min_price = np.min(prediction)

# Получение времени
now = datetime.now()
future_time = now + timedelta(hours=1)
future_time_str = future_time.strftime("%H:%M")

# Получение прогноза на следующий час
forecast_price = last_price + (last_price - df.iloc[-2]['price'])

# Получение времени прогноза на следующий час
forecast_time = last_row.name + pd.Timedelta(hours=1)

# Вывод результатов
print(f'Last price: {last_price:.2f} ({last_time})')
print(f'Forecast price: {forecast_price:.2f} ({forecast_time.strftime("%Y-%m-%d %H:%M:%S")})')

# Рисование графика
predictions = pd.read_csv('predictions.csv')
predictions.plot(x='date', y='price')
plt.show()
# data = scaler.inverse_transform(data)
# plt.plot(data, color='green')
# plt.plot(np.append(data, prediction), color='red')
# plt.title('Bitcoin Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price (USD)')
# date_form = DateFormatter("%H:%M")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(date_form)
# plt.show()

# постройте график с предсказанием на 1 час вперед
#plt.plot(data['date'], data['price'])
#df.plot(x='date', y='price')
#plt.plot(prediction['date'], prediction['price'])
#plt.show()
