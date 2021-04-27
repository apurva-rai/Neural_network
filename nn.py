from random import seed
from random import random
import numpy as np
import math

class NeuralNetwork:

    def __init__(self,width,depth,activate='r',iter=1000,rate=0.25):
        self.rate = rate
        self.iter = iter
        self.activate = activate
        self.width = width
        self.depth = depth

    #Calculates individual max for the np array and 0
    def individualMax(self,nn):
        return np.maximum(0,nn)

    #Calculates sigmoid function of input nn
    def sig(self,nn):
        return (1/(1+np.exp(-nn)))

    #Weights initalized
    def initalizeWeight(self, NNX):
        self.w_i = np.random.randn(NNX.shape[1],self.width) * np.sqrt(2/NNX.shape[1])
        self.b_i = np.random.randn(1, self.width)

        self.w = np.random.randn(self.depth, self.width, self.width) * np.sqrt(2/self.width)
        self.b = np.random.randn(self.depth, 1, self.width)

        self.w_n = np.random.randn(self.width,1) * np.sqrt(2/self.width)
        self.b_n = np.random.randn(1)

    def rSquare(self, NNY):
        self.score = 1 - (np.sum(np.square(self.finalValues-NNY))/np.sum(np.square(NNY-np.mean(NNY))))

    #Calculates intermediate values using the initialized weights and biases of the hidden layer
    def forward(self, NNX):
        self.v_i = np.dot(NNX,self.w_i) + self.b_i

        if self.activate == 'r':
            self.v_i = self.individualMax(self.v_i)
        elif self.activate == 'sig':
            self.v_i = self.sig(self.v_i)

        self.v = np.zeros((self.depth, NNX.shape[0], self.width))

        for i in range(self.depth):
            if i == 0:
                self.v[i] = np.dot(self.v_i, self.w[i]) + self.b[i]
            else:
                self.v[i] = np.dot(self.v[i-1], self.w[i]) + self.b[i]

            if self.activate == 'r':
                self.v[i] = self.individualMax(self.v[i])
            elif self.activate == 'sig':
                self.v[i] = self.sig(self.v[i])

        self.v_n = np.dot(self.v[self.depth-1],self.w_n) + self.b_n

    #Calculate the back propogation using the gradient of the weights and biases of the hidden layer
    def backward(self):
