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
import utils
import time

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
    n_X = X.shape[0]
    n_H = 4
    n_Y = Y.shape[0]
    
    return n_X, n_H, n_Y

    
# Initialize Model Parameters
def initialize_parameters(n_X, n_H, n_Y):
    """
    Arguments:
    n_X - size of input layer
    n_H - size of hidden layer
    n_Y - size of output layer
    Returns:
    params - Dictionary containing weights and biases
            W1 -- Weight matrix of shape (n_H, n_X)
            b1 -- Bias matrix of shape (n_H, 1)
            W2 -- Weight matrix of shape (n_Y, n_H)
            b2 -- Bias matrix of shape (n_Y, 1)
    """
    W1 = np.random.randn(n_H, n_X)*0.01
    b1 = np.zeros((n_H, 1))
    W2 = np.random.randn(n_Y, n_H)*0.01
    b2 = np.zeros((n_Y, 1))
    
    params = {"W1": W1, 
              "b1": b1,
              "W2": W2,
              "b2": b2}
    return params

# Forward Propagation
def forward_propagation(X, params):
    """
    Args:
    X -- Input data of shape (n_x, m) n_x -> no of input nodes; m -> no of training examples
    params -- Dictionary with the initialized weights and the biases
    Returns:
    A2 -- Sigmoid output of 2nd or output layer of shape (n_y, m)
    neuron_functions -- Dictionary containing the computed linear function Z and activation function A for each neuron
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    # Compute the linear function Z and sigmoid Activation function for each neuron
    Z1 = np.dot(W1,X) + b1
    A1 = utils.sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = utils.sigmoid(Z2)
    
    neuron_functions = {"Z1": Z1,
                        "A1": A1,
                        "Z2": Z2,
                        "A2": A2}
    return A2, neuron_functions

# Compute cost function
def compute_cost(A2, Y):
    """
    Args:
    A2 -- Predicted output value using sigmoid activation of shape (n_y, m) 
    Y -- Labels of shape (n_y, m)
    
    Returns:
    cost -- Average of cross entropy cost over m training examples
    """
    m = Y.shape[1] # number of training examples

    logprob = Y*np.log(A2) + (1-Y)*np.log(1-A2)
    cost = -(1/m)*np.sum(logprob)

    return cost

# Backward Propagation
def backward_propagation(neuron_functions, params, X, Y):
    """
    Args:
    neuron_functions -- Dictionary of neuron linear and action functions computed 
                                                        during forward propagation
    Returns:
    gradients - Dictionary with gradients of Loss w.r.t the weights and biases 
    """
    m = X.shape[1]
    
    W1 = params["W1"]
    W2 = params["W2"]
    A1 = neuron_functions["A1"]
    A2 = neuron_functions["A2"]

    dZ2 = A2 - Y # (n_y, m)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1, X.T) 
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return gradients

def update_parameters(params, gradients, learning_rate=1.2):
    """
    Updates parameters using gradient descent update rule
    
    Args:
    params -- Dictionary containing weights and biases
            W1 -- Weight matrix of shape (n_H, n_X)
            b1 -- Bias matrix of shape (n_H, 1)
            W2 -- Weight matrix of shape (n_Y, n_H)
            b2 -- Bias matrix of shape (n_Y, 1)
    gradients -- Dictionary with gradients of Loss w.r.t the weights and biases 
    learning_rate - Learning rate for gradient descent
    Returns:
    params -- Updated weights and biases 
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]    

    # Update parameters
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2

    params = {"W1": W1, 
            "b1": b1,
            "W2": W2,
            "b2": b2}
    
    return params

def nn_model_train(X, Y, num_iterations = 10000, print_cost=False, print_cost_itr=1000):
    """
    Train the Neural Network
    
    Args:
    X -- Input dataset
    Y -- Labels
    num_iterations -- Number of training iterations
    print_cost -- Set True to print cost during training
    print_cost_itr -- Print cost after every "print_cost_itr" iterations

    Returns:
    params -- Learned weights and biases after training
    """
    # Get NN layer sizes
    n_X, n_H, n_Y = layer_sizes(X, Y)
    print("Size of I/P layer: {}, hidden layer: {}, O/P layer: {}".format(n_X, n_H, n_Y))

    # Initialize weights and biases
    params = initialize_parameters(n_X, n_H, n_Y)
    print("Weights and biases matrix initialized as: {}".format(params))

    # Training loop
    for i in range(0, num_iterations):
        start_time = time.time()
        # Forward Propagation
        A2, neuron_functions = forward_propagation(X, params)
        # Cost function
        cost = compute_cost(A2, Y)
        # Backpropagation
        gradients = backward_propagation(neuron_functions, params, X, Y)
        # Gradient descent update weights and biases
        params = update_parameters(params, gradients)

        # Print cost every "print_cost_itr" iterations
        if print_cost and i%print_cost_itr == 0:
            print('Cost after {} iterations: {}'.format(i, cost))

    end_time = time.time()

    training_time = start_time-end_time
    print('Training time: ', training_time)

    return params






def main():
    X, Y = load_dataset()
    print('Loaded the planar flower dataset.')
    print('Input features: ', X)
    print('')
    print('Labels Red = 0, Blue = 1: ', Y)
    # Visualize the dataset
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    plt.show()
    params = nn_model_train(X,Y,print_cost=True)
    print('Learned weights and biases: ', params)


if __name__ ==  '__main__':
    main()
