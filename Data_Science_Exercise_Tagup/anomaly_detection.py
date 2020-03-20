
#import necessary libraries
import keras
import os
import h5py
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
from sklearn.externals.joblib import dump, load
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


class automated_prediction:

    def __init__(self, input_data, window_size, time_steps, scaler_filepath, lstm_filepath, thres_filepath):

        self.input_data = input_data
        self.window_size = window_size
        self.time_steps = time_steps
        self.scaler_filepath = scaler_filepath
        self.lstm_filepath = lstm_filepath
        self.thres_filepath = thres_filepath

    '''
    This automated method is aimed at making predictions from real world data based on the threshold obatined
    from the trained model which is an anomaly score to detect anomalies.
    The outputs are dataframe containing the times of anomaly and the anomalies plot.


    Parameters:
    input_data = input data in csv format
    window = window size for Moving Window Average
    time_steps = Number of timesteps for the LSTM Sequence
    sc_filepath = the trained scaler model, saved as ".bin"
    lstm_filepath = the trained LSTM model, saved as ".h5"
    threshold_filepath = the threshold as ".npy"
    ano_filepath = anomaly CSV file
    '''


    def LSTM_prediction(self):

        #load in the data
        df_machine = pd.read_csv(self.input_data, index_col = 0)

        #reformat your dataframe into datetime format
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
                dict_col["smooth_data_" + columns] = pd.Series(df[columns]).rolling(min_periods = 1, window = window).mean()

            df = pd.DataFrame(dict_col, index = df.index)
            return df

        df_machine.index = index_to_date(df_machine)
        df_smooth = moving_window_average(df_machine, self.window_size)

        ##Lets load in the models
        #return the scaler model
        scaler_model = load(self.scaler_filepath)
        # Returns a compiled model identical to the previous one
        lstm_model = load_model(self.lstm_filepath)
        #return the threshold
        threshold = np.load(self.thres_filepath)

        ##preprocess the data
        def preprocess(df, scaler_model):

            scaled_data = scaler_model.transform(df)

            #we converted back our data to df simply because we will still need in later sections
            df_scaled = pd.DataFrame(scaled_data, index = df.index)

            return scaled_data, df_scaled

        scaled_data, df_scaled = preprocess(df_smooth, scaler_model)

        def LSTM_3D_Input(arr, df, time_steps):

            '''
            This function takes in a dataframe that is intended to be used as LSTM Input data
            and transform the data from 2D input structure(Samples, features) to
            3D input structure [samples, time_steps, n_features].
            This is required for all LSTM Model structure in Keras.
            '''
            lstm_input = []
            for i in range(len(arr) - time_steps):
                v = arr[i:(i + time_steps)]
                lstm_input.append(v)

            lstm_input = np.array(lstm_input)

            #we need to define the data frame from where the timesteps began
            df_lstm = df[time_steps:]

            return lstm_input, df_lstm

        lstm_input, df_lstm = LSTM_3D_Input(scaled_data, df_scaled, self.time_steps)

        def flatten_to_2d(df):

            '''
            This flattens back the 3D LSTM output to the normal 2D output
            which is required for obtain our reconstruction error.
            X: A 3D array from lstm with shape == [sample, timesteps, features].
            flat_X:  A 2D array, sample x features.
            '''
            flat_lstm = np.empty((df.shape[0], df.shape[2]))  # obtain sample-->[0] and features only-->[2].
            for i in range(df.shape[0]):
                flat_lstm[i] = df[i, (df.shape[1]-1), :]

            return flat_lstm

        #make predictions and
        #generate the Mean Absolute Error (MAE) i.e. predicted value - actual value

        lstm_pred = lstm_model.predict(lstm_input)
        mae_loss = np.mean(np.abs(flatten_to_2d(lstm_pred) - flatten_to_2d(lstm_input)), axis = 1)

        def reconstructed_df(mae_loss, df, threshold):

            '''
            This basically creates a dataframe with columns that consist of
            the Mae_loss, threshold and anomaly.
            '''

            df_t = pd.DataFrame(index=df.index)
            df_t['Loss_mae'] = mae_loss
            df_t['Threshold'] = threshold
            df_t['Anomaly'] = df_t['Loss_mae'] > df_t['Threshold']
            return df_t

        df_machine = reconstructed_df(mae_loss, df_lstm, threshold)
        #Dataframe for times of fault and failures
        df_anomaly = df_machine[df_machine['Anomaly'] == True]
        
        #Lets plot times of faults and failures
        plt.plot(df_anomaly.index, df_anomaly.Loss_mae)
        plt.scatter(df_anomaly.index, df_anomaly.Loss_mae, c = 'red')
        plt.title('Times of fault and failures in this machine')
        plt.xticks(rotation = 45)
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, labels = ['Signals', 'Threshold'])
        
        return df_anomaly
