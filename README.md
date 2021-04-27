# Neural_network
Neural network implementation from scratch following a regression model

## Datasets Examined
Acheive relatively successful classification
Wheat Seeds Dataset - https://archive.ics.uci.edu/ml/datasets/seeds

Boston Housing Dataset - https://www.kaggle.com/puxama/bostoncsv

## The Algorithm

Since the implementation is mostly barebones using popular references it is not very optimized for performance or speed. The netword trains itself using a random initialization of weights and biases. The mean squared error is then calculated and backpropogation applied.

A simple K-fold validation is also implemented to verify the integrity of that run.

## Project

To run the project simply traverse to the directory with the project in it and type 'py main.py'. The project uses basic ML dependancies like numpy.

Since the algorithm is relatively slow I simply ran it for 10,000 and 100,000 iterations and there is a sharp increase in accuracy. It is not far fetched to be able to increase >0.9 r squared value with more iterations.

###### 10,000 iterations:

![alt text](https://github.com/apurva-rai/Neural_network/blob/main/images/run1.png?raw=true)

###### 100,000 iterations:

![alt text](https://github.com/apurva-rai/Neural_network/blob/main/images/run2.png?raw=true)

## Future work

I would want to optimize many of the processes in this implementation along with the libraries that I use. I would like to use jit Numba for certain tasks that are called often.
