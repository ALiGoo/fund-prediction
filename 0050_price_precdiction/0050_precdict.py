# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('/home/xiangli/Programming/0050_price_precdiction/0050_train.csv')
training_set = dataset_train.iloc[:, 0:1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 16))
#regressor.add(Dropout(0.2))

# =============================================================================
# # Adding a fifth LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))
# =============================================================================

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('/home/xiangli/Programming/0050_price_precdiction/0050_test.csv')
real_stock_price = dataset_test.iloc[:, 0:1].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['nav'], dataset_test['nav']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 160):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.figure(figsize=(20,15))
plt.plot(pd.DataFrame(real_stock_price).pct_change().iloc[1:].values, color = 'red', label = 'Real 0050 Stock Price')
plt.plot(pd.DataFrame(predicted_stock_price).pct_change().iloc[1:].values, color = 'blue', label = 'Predicted 0050 Stock Price')
plt.title('0050 Stock Price Prediction', fontsize=20)
plt.xlabel('Time', fontsize=20)
plt.ylabel('0050 Stock Price', fontsize=20)
plt.legend(fontsize=20)
plt.savefig("/home/xiangli/Programming/0050_price_precdiction/0050_precdict4.png")
plt.show()
