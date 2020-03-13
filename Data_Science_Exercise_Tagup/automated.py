#import necessary libraries
import keras
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt

from numpy.random import seed
from datetime import datetime

from keras import optimizers, Sequential
from keras.layers import Input, Dropout
from keras.layers.core import Dense
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json
import warnings
warnings.filterwarnings("ignore")

#set the data visualization parameters

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'figure.figsize': (16, 8),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


def automated_method(input_data, window, split_size, time_steps, learning_rate, epoch, batch_size):

    '''
    This automated method is aimed to pinpoint times of failures and faults in machines.
    The output will be a dataframe and a plot showing times of anomalies in the machine data set.

    Parameters:
    input_data = input data in csv format
    window = window size for Moving Window Average
    split_size = Split size should be between 0.00 and 1.00, for splitting the dataset into train and test data
    time_steps = Number of timesteps for the LSTM Sequence
    learning_rate = Learning rate for Adam Optimizer
    epoch = Number of Epochs
    batch_size = Batch size
    '''

    df_machine = pd.read_csv(input_data, index_col = 0)

    def index_to_date(df):

        '''
        This function convert the dataframe index to a Year-Month-Date Time format
        '''
        df.index = pd.to_datetime(df.index, format= '%Y-%m-%d %H:%M:%S.%f')
        df.index = df.index.map(lambda t: t.strftime('%Y-%m-%d'))
        return df.index

    #moving window average
    def moving_window_average(df, window):

        '''
        The moving average filter is a simple Low Pass FIR (Finite Impulse Response) filter,
        commonly used for smoothing an array of sampled data/signal.

        At a timestep, an arithmetic mean is taken from a specified number of data points to produce a singular output point.
        the number of the data points needed to compute an average for each time step is refered to as "window"
        The averaging window is moved over the data, shifting it by one time step after each calculation ("moving average").
        '''
        dict_col = {}
        for columns in df.columns:
            dict_col["smooth_data_" + columns] = pd.Series(df[columns]).rolling(min_periods = 1, window=window).mean()

        df_smooth = pd.DataFrame(dict_col, index = df.index)
        return df_smooth

    df_machine.index = index_to_date(df_machine)
    #denoise the data
    df_smooth = moving_window_average(df_machine,window)

    #Data splitting
    train_size = int(len(df_smooth) * split_size)
    test_size = int(len(df_smooth) - train_size)
    train = df_smooth.iloc[0:train_size]
    test = df_smooth.iloc[train_size:len(df_smooth)]

    #Data preprocessing
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    #we converted back our data to df simply because we will still need in later sections
    df_train = pd.DataFrame(train_scaled, index = train.index)
    df_test = pd.DataFrame(test_scaled, index = test.index)

    def LSTM_3D_Input(df, time_steps):

        '''
        This function takes in a dataframe that is intended to be used as LSTM Input data
        and transform the data from 2D input structure(Samples, features) to
        3D input structure [samples, time_steps, n_features].
        This is required for all LSTM Model structure in Keras.
        '''
        X_3D = []
        for i in range(len(df) - time_steps):
            v = df[i:(i + time_steps)]
            X_3D.append(v)

        return np.array(X_3D)

    x_train = LSTM_3D_Input(train_scaled, time_steps)
    x_test = LSTM_3D_Input(test_scaled, time_steps)

    df_train_lstm = df_train.iloc[time_steps:]
    df_test_lstm = df_test.iloc[time_steps:]

    def flatten_to_2d(X):

        '''
        This flattens back the 3D LSTM output to the normal 2D output
        which is required for obtain our reconstruction error.

        X: A 3D array from lstm with shape == [sample, timesteps, features].
        flat_X:  A 2D array, sample x features.
        '''
        flat_X = np.empty((X.shape[0], X.shape[2]))  # obtain sample-->[0] and features only-->[2].
        for i in range(X.shape[0]):
            flat_X[i] = X[i, (X.shape[1]-1), :]

        return(flat_X)

    #LSTM parameters
    adam = optimizers.Adam(learning_rate)

    #Building the LSTM model itself using KERAS
    lstm_autoencoder = Sequential()
    # Encoder
    lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(x_train.shape[1]))
    # Decoder
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(x_train.shape[2])))

    lstm_autoencoder.compile(loss='mse',optimizer=adam)

    #train the model
    lstm_autoencoder_history = lstm_autoencoder.fit(
    x_train, x_train,
    epochs=epoch,
    batch_size=batch_size,
    validation_split=0.1,
    shuffle=False)

    #make predictions and #generate the Mean Absolute Error (MAE) i.e. predicted value - actual value
    x_pred = lstm_autoencoder.predict(x_train)
    train_mae_loss = np.mean(np.abs(flatten_to_2d(x_pred) - flatten_to_2d(x_train)), axis = 1)

    #Threshold - represent the maximum value of MAE for a machine in "Normal Mode", any higher than this indicates otherwise
    #i.e. faulty or failed
    threshold = max(train_mae_loss).round(2)

    #make predictions and #generate the Mean Absolute Error (MAE) for test data
    x_pred_test = lstm_autoencoder.predict(x_test)
    test_mae_loss = np.mean(np.abs(flatten_to_2d(x_pred_test) - flatten_to_2d(x_test)), axis = 1)

    def reconstructed_df(mae_loss, df, threshold):

        '''
        This basically creates a dataframe with columns that consist of the Mae_loss, threshold and anomaly.
        '''
        df_t = pd.DataFrame(index=df.index)
        df_t['Loss_mae'] = mae_loss
        df_t['Threshold'] = threshold
        df_t['Anomaly'] = df_t['Loss_mae'] > df_t['Threshold']
        return df_t

    df_recon_train = reconstructed_df(train_mae_loss, df_train_lstm, threshold)
    df_recon_test = reconstructed_df(test_mae_loss, df_test_lstm, threshold)
    df_machine = pd.concat([df_recon_train, df_recon_test])
    df_anomaly = df_machine[df_machine['Anomaly'] == True]

    #plot the anomaly times
    plt.plot(df_machine.index, df_machine.Loss_mae)
    plt.scatter(df_anomaly.index, df_anomaly.Loss_mae, c = 'red')
    plt.xticks([0, 200, 400, 600, 800, 950])
    plt.xlabel('Time')
    plt.ylabel('Reconstruction Error')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, labels = ['Signals', 'Anomalies'])
    plt.title('Times of faults and failures (anomalies) in the Machine')
    plt.show()

    return df_machine
