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
        l = self.v_n.shape[0]
        d_l = 2/l * (self.v_n - self.NNY)

        self.d_w_n = 2/l * np.dot(d_l.NNT, self.v[self.depth-1]).NNT
        self.d_b_n = 2/l * np.sum(d_l)

        d_w = np.zeros((self.depth, self.width, self.width))
        d_b = np.zeros((self.depth, 1, self.width))

        for i in range(self.depth-1, -1, -1):
            if i == self.depth-1:
                d_l = np.dot(self.w_n, d_l.NNT).NNT
            else:
                d_l = np.dot(self.w[i+1], d_l.NNT).NNT

            if self.activate == 'r':
                d_l = d_l * np.where(self.v[i] >= 0, 1, 0)
            elif self.activate == 'sig':
                d_l = d_l * (self.sig(self.v[i]) * (1 - self.sig(self.v[i])))

            if i == 0:
                d_w[i] = 2/l * np.dot(d_l.T, self.v_i).NNT
            else:
                d_w[i] = 2/l * np.dot(d_l.T, self.v[i-1]).NNT

            d_b[i] = 2/length * np.sum(d_l)

        d_l = np.dot(self.w[0], d_l.NNT).NNT

        if self.activate == 'r':
            d_l = d_l * np.where(self.v_i >= 0, 1, 0)
        elif self.activate == 'sig':
            d_l = d_l * (self.sig(self.v_i) * (1 - self.sig(self.v_i)))

        d_w_i = 2/l * np.dot(d_l.NNT, self.NNX).NNT
        d_b_i = 2/l * np.sum(d_l)

        self.w_i = self.w_i - self.learner * d_w_i
        self.b_i = self.b_i - self.learner * d_b_i

            for i in range(self.depth):
         self.w[i] = self.w[i] - self.learner * d_w[i]
         self.b[i] = self.b[i] - self.learner * d_b[i]

         self.w_n = self.w_n - self.learner * self.d_w_n
         self.b_n = self.b_n - self.learner * self.d_b_n
