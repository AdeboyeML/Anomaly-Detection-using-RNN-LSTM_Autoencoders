## Anomaly-Detection-using-RNN-LSTM-Autoencoders

ExampleCo, Inc is gathering several types of data for its fleet of very expensive machines.  These very expensive machines have three operating modes: *normal*, *faulty* and *failed*.   The machines run all the time, and usually they are in normal mode.  However, in the event that the machine enters faulty mode, the company would like to be aware of this as soon as possible.  This way they can take preventative action to avoid entering failed mode and hopefully save themselves lots of money.

They collect four kinds of timeseries data for each machine in their fleet of very expensive machines.  When a machine is operating in *normal* mode the data behaves in a fairly predictable way, but with a moderate amount of noise.  Before a machine fails it will ramp into *faulty* mode, during which the data appears visibly quite different.  Finally, when a machine fails it enters a third, and distinctly different, *failed* mode where all signals are very close to 0.

You can download the data here: [exampleco_data](https://drive.google.com/open?id=1b12u6rzkG1AxB6wLGl7IBVoaoSoZLHNR)

### Main objective: to develop an automated method to pinpoint the times of fault and failure in this machine.  


### Steps Taken to accomplish the task:

1. Load in Data and necessary libraries
2. Data exploration
3. Noise Removal using Moving Window Average
4. Data Splitting
5. Data preparation
6. Implementation of LSTM Autoencoder
7. Data visualization to pinpoints times of fault and failures (~ to detect anomalies).


### Building and Implementation of the LSTM Autoencoder Model
#### Brief overview of LSTM Autoencoder model and the reason why we decided to use it
##### Autoencoder and Reconstruction Error:


Autoencoder: are neural networks that aim to reconstruct their input. They consist of two parts: an encoder and a decoder. The encoder maps input data to a latent space (or hidden representation) and the decoder maps back from latent space (~ hidden representation) to input space.

Typically the latent space has a lower dimensionality than the input space and, hence, Autoencoders are forced to learn compressed representations of the input data which enables them to captures the correlations and interactions between the input data.
The autoencoder is trained to reconstruct data with normal pattern (e.g., normal time series) by minimizing a loss function that measures the quality of the reconstructions. After training, the model can now reconstruct the normal data well enough with minimal reconstruction error.

Once training is completed, the reconstruction error is used afterwards as an anomaly score to detect anomaly in future time instance of the data i.e. if the model is given an anomalous sequence of the data in future time that is NOT NORMAL, it may not be able to reconstruct it well and hence would lead to higher reconstruction error compared to the resconstruction errors for the normal sequence. This is reason why we have to assume that the training data is said to be in NORMAL STATE.


#### LSTM Autoencoder and reason why we used it;


**LSTM (Long Short Term Memory)** is an upgraded Recurrent Neural Network (RNN), a powerful sequence learner with a memory cell and gates that control the information to include, remove and output from the memory cell. The major attribute of LSTM in comparision to RNN is the memory cell that stores long to short term information about input sequence across the timesteps.


In our case, LSTM Autoencoder will be used for sequence to sequence (seq2seq) learning i.e. the encoder reads a variable-length input sequence and converts it into a fixed-length vector representation (reduced dimension), and the decoder takes this vector representation and converts it back into a variable length sequence.


In general, the learned vector representation corresponds to the final hidden state of the encoder network, which acts like a summary of the whole sequence. Our LSTM Autoencoder is an example of Seq2Seq autoencoder, in which the input and output sequences are aligned in time (x = y) and, thus, have equal lengths (Tx = Ty).


**Reason we used LSTM Autoencoder:**


We used LSTM Autoencoder simply because our time series data is a sequential data and LSTM captures the temporal dependencies of the data by introducing memory.
Specifically, LSTM has the ability to capture long term temporal interactions and correlations between variables in the input sequence which is highly required in this scenario since this relationship are time dependent and they determine the state of the machine.





### Summary:

I believe the predictions are good because the model takes into account the behaviour of the signals over time and the error shows consistent decrease with increasing epochs.

Based on our results, LSTM Autoencoder is a robust model for detecting anomalies in time-series data, this is because it takes into account the temporal dependencies of the input sequence.

More data is required for each single machine to be able to justify the findings of these results because the more data for a single machine the more robust the model for predicting times of anomalies for the machine.



### ***Tools utilized: Python libraries: pandas, numpy, keras, seaborn.***
