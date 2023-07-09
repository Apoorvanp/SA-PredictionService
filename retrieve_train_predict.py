import numpy as np
import os

def save_train_predict(trainPredict):
    np.save('train_predict.npy', trainPredict)

def retrieve_train_predict():
    if 'train_predict.npy' in os.listdir():
        trainPredict = np.load('train_predict.npy')
        return trainPredict
    else:
        return None
