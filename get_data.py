import requests
import pandas as pd
import time

url = 'https://api.binance.com/api/v3/ticker/price'
params = {'symbol': 'BTCUSDT'}
csv_file = 'data.csv'
i = 1
count = 0

while i <= 20000:
    try:
        # Запрос данных с помощью API
        response = requests.get(url, params=params)
        response.raise_for_status()  # Вызов исключения, если запрос не был успешным
        data = response.json()

        # Добавление столбца с текущей датой и временем
        now = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
        data['date'] = now

        # Масштабирование цены
        data['price'] = "{:.3f}".format(float(data['price']))

        # Создание DataFrame из полученных данных
        df = pd.DataFrame(data, index=[0])

        # Добавление данных в CSV-файл
        with open(csv_file, 'a') as f:
            df.to_csv(f, header=f.tell()==0, index=False)

        #print(f"Значение {i}: Done.")

        # Увеличение счетчика итераций
        i += 1

        # Сброс счетчика ошибок
        count = 0

        # Задержка в 4 секунд между итерациями
        time.sleep(4)

    # `Если проблемы с ответом от сервера или с интернетом обрабатываем ошибки и ждем 10 секунд для повторного запроса
    except requests.exceptions.RequestException as e:
        count += 1
        print(f"Получаем значение {i}: {e}")
        print("Попробуем еще раз через 10 секунд...")
        time.sleep(10)
        if count >= 10:
            print(f"Потеряна связь с сервером. Полученно {i-1} значений.")
            break
print("Готово.")
