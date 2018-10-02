import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import SGD, Adam
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
# from keras.models import load_model

# from keras import backend as K
# from keras import backend as K
# import tensorflow as tf

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot

# import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
# from util.multi_gpu2 import make_parallel

EPOCHS = 500
BATCH_SIZE = 2056
# BATCH_SIZE = 2056*2
N_INPUT = 100
N_OUTPUT = 100
N_HIDDEN = 256

N_CONTEXT = 20 # The number of frames in the context


def train(file_name):
    X = np.load('data/npy/X.npy')
    Y = np.load('data/npy/Y.npy')

    N_train = int(len(X)*0.9)
    N_validation = len(X) - N_train

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)
    print(X.shape)
    print(Y.shape)
    print(X_train.shape)
    print(Y_train.shape)


    model = Sequential()
    model.add(TimeDistributed(Dense(N_HIDDEN), input_shape=(N_CONTEXT, N_INPUT)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Bidirectional(SimpleRNN(N_HIDDEN, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(N_OUTPUT))
    model.add(Activation('linear'))

    print (model.summary())

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


    model.compile(loss="mean_squared_error", optimizer=optimizer)
    
    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_validation, Y_validation))


    model.save(file_name)
        
    pyplot.plot(hist.history['loss'], linewidth=3, label='train')
    pyplot.plot(hist.history['val_loss'], linewidth=3, label='valid')
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.savefig('losses.png')

if __name__ == "__main__":
    N_CONTEXT = int(sys.argv[2])
    print('sys.argv[1]')
    train(sys.argv[1])
