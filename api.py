import csv
import math
from pylab import rcParams
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Conv2D, Conv1D, MaxPooling3D, MaxPooling2D
from keras.layers import Flatten, LSTM, Lambda, Reshape, BatchNormalization, GRU, AveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import keras.optimizers as ops

from flask import Flask, jsonify

app = Flask(__name__)


def fixData(df):
    df = df.drop(df.index[[0, 1, 2]])
    df = df.drop('MW', axis=1)
    df.columns = ['Date', 'From', 'to', 'MW']

    dates = df['Date'].tolist()

    days = []
    months = []
    years = []
    for date in dates:
        days.append(date.split('.')[0])
        months.append(date.split('.')[1])
        years.append(date.split('.')[2])

    df['Day'] = days
    df['Month'] = months
    df['Year'] = years

    def fixMW(cols):
        TT = cols
        TT = TT.replace('.', '')
        TT = TT.replace(',', '.')
        return TT

    df['MW'] = df['MW'].apply(fixMW).astype(float)
    df['Day'] = df['Day'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)

    df = df.groupby(['Year', 'Month', 'Day'])['MW'].sum()
    df = pd.DataFrame(df)
    df['Month'] = df.index.get_level_values('Month')
    df['Day'] = df.index.get_level_values('Day')
    df['Year'] = df.index.get_level_values('Year')
    df.reset_index(drop=True, inplace=True)
    return df


def Xplot(hist):
    f, axarr = plt.subplots(1, 1, figsize=(20, 5))

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    axarr.set_title('Model loss')
    axarr.set_ylabel('Loss')
    axarr.set_xlabel('Epoch')
    axarr.legend(['Train', 'val'], loc='upper left')
    plt.show()


def visualiser(df):
    f, axarr = plt.subplots(1, 1, figsize=(20, 5))
    df = df.drop('Month', axis=1)
    df = df.drop('Day', axis=1)
    df = df.drop('Year', axis=1)
    df = df.values
    df = df.astype('float32')
    plt.plot(df)
    plt.show()
    return df


def split1(df):
    train_size = int(len(df) * 0.75)
    test_size = len(df) - train_size
    train, test = df[0:train_size, :], df[train_size:len(df), :]
    return train, test


def split2(df, look_back=7):
    dataX, dataY = [], []
    for i in range(len(df)-look_back-1):
        a = df[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(df[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


Data1 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2010.csv', sep=';')
Data2 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2011.csv', sep=';')
Data3 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2012.csv', sep=';')
Data4 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2013.csv', sep=';')
Data5 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2014.csv', sep=';')
Data6 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2015.csv', sep=';')
Data7 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2016.csv', sep=';')
Data8 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2017.csv', sep=';')
Data9 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2018.csv', sep=';')
Data10 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2019.csv', sep=';')

Data1 = fixData(Data1)
Data2 = fixData(Data2)
Data3 = fixData(Data3)
Data4 = fixData(Data4)
Data5 = fixData(Data5)
Data6 = fixData(Data6)
Data7 = fixData(Data7)
Data8 = fixData(Data8)
Data9 = fixData(Data9)
Data10 = fixData(Data10)

Data = pd.concat([Data1, Data2, Data3, Data4, Data5, Data6,
                 Data7, Data8, Data9, Data10], ignore_index=True)

Data.head()

Data[(Data['Month'] == 1) & (Data['Year'] == 2010)]

Data0 = Data
Data0 = visualiser(Data0)

scaler = MinMaxScaler(feature_range=(0, 1))
Data0 = scaler.fit_transform(Data0)

train, test = split1(Data0)

trainX, trainY = split2(train)
testX, testY = split2(test)

trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

model = Sequential()
model.add(GRU(units=8, input_shape=(
    trainX.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))

model.add(GRU(units=16))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=ops.Nadam(lr=1e-3), loss='mse')

model.summary()
history = model.fit(trainX, trainY, validation_split=0.2,
                    epochs=100, batch_size=8, verbose=1)

Xplot(history)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

scaler2 = MinMaxScaler()
scaler2.min_, scaler2.scale_ = scaler.min_[0], scaler.scale_[0]

trainPredict = scaler2.inverse_transform(trainPredict)
trainYp = scaler2.inverse_transform([trainY])
testPredict = scaler2.inverse_transform(testPredict)
testYp = scaler2.inverse_transform([testY])

rcParams['figure.figsize'] = 20, 5


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainYp[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testYp[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(Data0)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[7:len(trainPredict)+7, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(Data0)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(7*2)+1:len(Data0)-1, :] = testPredict
# plot baseline and predictions

plt.plot(scaler.inverse_transform(Data0), color='y')
plt.plot(trainPredictPlot, color='b')
plt.plot(testPredictPlot, color='r')
plt.show()






@app.route('/')
def hello():
    return 'Welcome to the Solar Energy Forecasting API!'


@app.route('/prediction/hourly')
def get_hourly_prediction():
    start_time = ''
    end_time = ''
    with open('21_01_2019.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            date = row[0]
            time = row[1]
            amount = int(row[3])
            if amount > 20:
                if not start_time:
                    start_time = time
                    end_time = time
                else:
                    end_time = time
            elif start_time:
                break

    prediction_result = {
        'start_time': start_time,
        'end_time': end_time,
        'prediction': 'Hourly prediction result is here'
    }

    return jsonify(prediction_result)




@app.route('/prediction/monthly')
def get_monthly_prediction():
    # Perform monthly prediction using your trained model
    # Assuming you have already performed the prediction and stored the result in the variable `trainPredict`

    column_names = ['MW Total']
    df = pd.DataFrame(trainPredict, columns=column_names)

    high_production_days = df.nlargest(5, 'MW Total')['MW Total'].index.tolist()
    low_production_days = df.nsmallest(5, 'MW Total')['MW Total'].index.tolist()

    response = {
        'prediction': 'Monthly prediction result',
        'high_production_days': high_production_days,
        'low_production_days': low_production_days
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run()
