import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout



dataset = pd.read_csv(r"D:\test\BTC-USD Training Data - 1st Jan 2016 to 1st Jan 2022.csv")

dataset.info()

dataset["Close"]=pd.to_numeric(dataset.Close,errors='coerce')
dataset=dataset.dropna()
traindata = dataset.iloc[:,4:5].values

dataset.info()

sc = MinMaxScaler(feature_range=(0,1))
traindata = sc.fit_transform(traindata)
traindata.shape

x_train = []
y_train = []

for i in range (60,1149):
    x_train.append(traindata[i-60:i,0])
    y_train.append(traindata[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

model = Sequential()

model.add(LSTM(units=100, return_sequences= True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences= True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences= True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences= False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')

hist= model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=2)

plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

testdata = pd.read_csv(r"D:\test\BTC-USD Out of Time Testing 1st Jan 2022 to 4th Feb 2022.csv")
testdata['Close']=pd.to_numeric(testdata.Close,errors='coerce')
testdata = testdata.dropna()
testdata = testdata.iloc[:,4:5]
y_test = testdata.iloc[60:,0:].values

inputClosing = testdata.iloc[:,0:].values
inputClosing_scaled = sc.fit_transform(inputClosing)
inputClosing_scaled.shape
x_test = []
length = len(testdata)
timestep = 60
for i in range(timestep,length):
    x_test.append(inputClosing_scaled[i-timestep:i,0])

x_test = np.array(x_test)


x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape
y_pred = model.predict(x_test)
y_pred

predicted_price = sc.inverse_transform(y_pred)

plt.plot(y_test, color = 'red', label ='Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('NSE Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


