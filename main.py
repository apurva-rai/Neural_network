import numpy as np
import sys
import NeuralNetwork as nn

NNX = np.loadtxt('data/wheat-seeds.csv',delimiter=',')
NNY = NNX[:,-1:]
NNX = NNX[:, :=1]

model1 = nn.nn(13,3,actiavte='r',iter=1000,rate=0.01)
print('\nSeed Dataset:\nTraining r square score: '+ str(model.initializeNN(NNX,NNY)) + '\n25-fold cross validation r square score: ' + str(model.trainNN(NNX,NNY)))
