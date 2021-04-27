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
    def initalizeWeight(self, nn):
        self.w_i = np.random.randn(nn.shape[1],self.width) * np.sqrt(2/nn.shape[1])
        self.b_i = np.random.randn(1, self.width)

        self.w = np.random.randn(self.depth, self.width, self.width) * np.sqrt(2/self.width)
        self.b = np.random.randn(self.depth, 1, self.width)

        self.w_n = np.random.randn(self.width,1) * np.sqrt(2/self.width)
        self.b_n = np.random.randn(1)
