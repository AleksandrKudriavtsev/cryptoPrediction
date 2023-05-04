import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Загружаем данные из файла data.csv
df = pd.read_csv('data.csv')

# Преобразуем столбец 'date' в тип datetime и устанавливаем его в качестве индекса
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Рисуем график цены биткоина по времени
#df['price'].plot()
#plt.show()

# Нормализуем данные в диапазон от 0 до 1
scaler = MinMaxScaler()
data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

# Задаём размер окна
window_size = 200

# Разделяем данные на обучающую, валидационную и тестовую выборки (70%, 15%, 15%)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
test_size = len(data) - train_size - val_size
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

# Создаём набор данных для временных рядов с заданным размером окна
def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i])
        y.append(dataset[i])
    return np.array(X), np.array(y)

# Создаем наборы данных для обучения, валидации и тестирования с заданным window size

X_train, y_train = create_dataset(train_data, window_size)
X_val, y_val = create_dataset(val_data, window_size)
X_test, y_test = create_dataset(test_data, window_size)

# Создаем и обучаем LSTM-модель с расчётом потерь MSE
model = tf.keras.Sequential([
tf.keras.layers.LSTM(10, input_shape=(window_size, 1)),
tf.keras.layers.Dropout(0.2), # Добавляем дропаут для предотвращения переобучения
tf.keras.layers.Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_val, y_val), callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]) # Добавляем раннюю остановку для предотвращения переобучения

# Предсказываем цену биткоина на тестовой выборке и обратно преобразуем данные в исходный масштаб
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Вычисляем среднеквадратичную ошибку (MSE) среднюю абсолбтную ошибку (MАE) и корень из среднеквадратичной ошибки (RMSE) на тестовой выборке
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# Рисуем график фактической и прогнозируемой цены биткоина на тестовой выборке
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

# Подготавливаем данные для прогнозирования на час вперед с использованием прямого метода
last_data = data[-window_size:]
last_data = last_data.reshape((1, window_size, 1))

# Предсказываем цену биткоина на час вперед с шагом в 4 секунды (время взял произвольное) с использованием прямого метода
num_predictions = int((60 * 60) / 4) # Количество прогнозов на час вперед с шагом в 4 секунды
future_data = np.repeat(data[-window_size:], num_predictions, axis=0) # Создаем массив из повторяющихся последних данных
future_time = np.arange(1, num_predictions + 1).reshape(-1, 1) # Создаем массив из последовательных значений времени
future_data = future_data.reshape((num_predictions, -1)) # Изменяем размерность массива future_data на (900, 11)
future_data = np.concatenate([future_data, future_time], axis=1) # Объединяем массивы данных и времени
future_data = np.expand_dims(future_data, axis=2) # Добавляем измерение признака и изменяем размерность массива future_data на (900, 1, 12)

#Получаем длину массива future_data (что бы при подборе размера окна избежать несовпадения размера массивов)
n = len(future_data)
#Получаем остаток от деления длины массива на размер окна
r = n % window_size
#Если остаток не равен нулю, обрезаем массив на остаток
if r != 0:
    future_data = future_data[:-r]
    
future_data = future_data.reshape((-1, window_size, 1)) # Добавляем измерение окна и учитываем размер окна при создании батчей
predictions = model.predict(future_data) # Предсказываем цену для каждого значения времени
predictions = scaler.inverse_transform(predictions) # Обратно преобразуем данные в исходный масштаб

# Получаем последнее значение времени из исторических данных
last_date = df.index[-1]

# Генерируем новые значения дат и времени для прогнозируемых цен
predicted_dates = pd.date_range(last_date + pd.Timedelta(seconds=4), periods=num_predictions - window_size + 1, freq='4S') # Учитываем размер окна при генерации дат

# После создания массивов predictions и predicted_dates
print (len (predictions)) # Печатает длину массива predictions, например 893
print (len (predicted_dates)) # Печатает длину массива predicted_dates, например 897

# Если длины не совпадают, вы можете сделать одно из следующего:
predictions = predictions [:len (predicted_dates)] # Обрезает массив predictions до длины predicted_dates
# Создаем новый DataFrame с прогнозируемыми ценами и датами
predictions = pd.DataFrame({'price': predictions.flatten(), 'date': predicted_dates}) # Изменяем форму predictions на одномерный массив

# Сохраняем прогноз в файл predictions.csv
predictions.to_csv('predictions.csv', index=False)

# Рисуем график прогнозируемых цен на час вперед
plt.plot(df.index, df['price'], label='Actual') # Добавляем график исторических цен для сравнения
plt.plot(predicted_dates, predictions['price'], label='Predicted') # Добавляем метку к графику прогнозных цен
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Prediction for Next Hour')
plt.legend() # Добавляем легенду к графику
plt.show()
