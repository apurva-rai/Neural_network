import numpy as np
import sys
from nn import NeuralNetwork
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    NNX = np.loadtxt('data/wheat-seeds.csv',delimiter=',')
    NNY = NNX[:,-1:]
    NNX = NNX[:, :-1]

    model1 = NeuralNetwork(10,3,activate='r',iter=1000,rate=0.1)
    print('\nSeed Dataset:\nTraining r square score: '+ str(model1.initializeNN(NNX,NNY)) + '\n15-fold cross validation r square score: ' + str(model1.trainNN(NNX,NNY)))

    NNX = np.loadtxt('data/bikes.csv',delimiter=',')
    NNY = NNX[:,-1:]
    NNX = NNX[:, :-1]

    model2 = NeuralNetwork(10,3,activate='r',iter=1000,rate=0.1)
    print('\nHousing Dataset:\nTraining r square score: '+ str(model2.initializeNN(NNX,NNY)) + '\n15-fold cross validation r square score: ' + str(model2.trainNN(NNX,NNY)))
