
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
from sklearn.externals import joblib
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


class automated_training:

    def __init__(self, input_data, window_size, time_steps, learning_rate, epoch, batch_size, sc_filepath, lstm_filepath, thres_filepath):

        self.input_data = input_data
        self.window = window_size
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.sc_filepath = sc_filepath
        self.lstm_filepath = lstm_filepath
        self.thres_filepath = thres_filepath

        '''
        This automated method is aimed at learning the temporal relationships between the mutlivariate features
        in this dataset.
        The outputs are saved scaler model, saved LSTM model and threshold, all of theses are
        required for future predictions.

        Parameters:
        input_data = input data in csv format
        window = window size for Moving Window Average
        time_steps = Number of timesteps for the LSTM Sequence
        learning_rate = Learning rate for Adam Optimizer
        epoch = Number of Epochs
        batch_size = Batch size
        sc_filepath = filepath for the trained scaler model, saved as ".bin"
        lstm_filepath = filepath for the trained LSTM model, saved as ".h5"
        threshold_filepath = saves the threshold as ".npy"
        '''

    def LSTM_training(self):

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
        df_smooth = moving_window_average(df_machine, self.window)

        #Data preprocessing
        def preprocess(df, filepath):

            scaler = StandardScaler()
            scaler.fit(df)
            scaled_data = scaler.transform(df)

            '''
            dump(scaler_model, filepath) --> Save the scaled model for Reuse on real world data sets
            This will save it in the current working directory or another directory if indicated
            '''
            joblib.dump(scaler, filepath, compress=True)

            #we converted back our data to df simply because we will still need in later sections
            df_scaled = pd.DataFrame(scaled_data, index = df.index)

            return scaled_data, df_scaled

        scaled_data, df_scaled = preprocess(df_smooth, self.sc_filepath)

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

        lstm_input, df_scaled_lstm = LSTM_3D_Input(scaled_data, df_scaled, self.time_steps)

        #Model Implementation and Threshold Generation
        def lstm_model(lstm_input, lr, epoch, batch_size, filepath):

            #LSTM parameters
            adam = optimizers.Adam(lr)

            #Building the LSTM model itself using KERAS
            lstm_autoencoder = Sequential()
            # Encoder
            lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(lstm_input.shape[1], lstm_input.shape[2]), return_sequences=True))
            lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
            lstm_autoencoder.add(RepeatVector(lstm_input.shape[1]))
            # Decoder
            lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
            lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
            lstm_autoencoder.add(TimeDistributed(Dense(lstm_input.shape[2])))
            lstm_autoencoder.compile(loss='mse',optimizer=adam)

            #train the model
            lstm_autoencoder_history = lstm_autoencoder.fit(lstm_input, lstm_input, epochs=epoch,
                                                        batch_size=batch_size,validation_split=0.1,
                                                        shuffle=False)


            pred_lstm = lstm_autoencoder.predict(lstm_input)

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

            #generate the Mean Absolute Error (MAE) i.e. predicted value - actual value
            df_mae_loss = np.mean(np.abs(flatten_to_2d(pred_lstm) - flatten_to_2d(lstm_input)), axis = 1)
            threshold = max(df_mae_loss).round(2)

            '''
            model.save(filepath) --> save the keras LSTM model
            The file path should be in a HDF5 file format 'my_model.h5'
            '''
            lstm_autoencoder.save(filepath)
            return threshold

        threshold = lstm_model(lstm_input, self.learning_rate, self.epoch, self.batch_size, self.lstm_filepath)
        threshold = np.array(threshold)

        '''
        np.save(filepath, np.array) --> This save the threshold in a npy file format,
        this will be needed for future predictions
        This is saved in the current directory or another directory if specified.
        '''
        np.save(self.thres_filepath, threshold)
    
