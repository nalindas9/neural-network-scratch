"""
Planar data classification using 1 hidden layer

Reference:
1. Coursera Neural Networks and Deep Learning

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Planar flower dataset definition
def load_dataset():
    """
    Arguments:
    None
    Returns:
    X - input feature dataset of shape (input size, number of examples)
    Y - labels of shape (output size, number of examples)
    """
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T
   
    return X, Y

# Define input, output and hidden layer size   
def layer_sizes(X, Y):
    """
    Arguments:
    X - input feature dataset of shape (input size, number of examples)
    Y - labels of shape (output size, number of examples)
    Returns:
    n_X - size of input layer
    n_H - size of hidden layer
    n_Y - size of output layer
    """
    print("X Shape: ", X.shape, "Y Shape: ", Y.shape)
    

def main():
    X, Y = load_dataset()
    print('Loaded the planar flower dataset.')
    print('Input features: ', X)
    print('')
    print('Labels Red = 0, Blue = 1: ', Y)
    # Visualize the dataset
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    plt.show()
    layer_sizes(X, Y)
    
if __name__ ==  '__main__':
    main()
