"""


Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import numpy as np

def initialize_parameters(layer_dims):
    """
    Initializes the weights and biases for all the layers in the NN
    Arguments:
    layer_dims -- List containing size of each layer in DNN
    
    Returns:
    params - Dictionary with the initialized weights and biases for each layer in DNN
    """
    
    num_layers = len(layer_dims)
    params = {}
    for i in range(1, num_layers):
        params['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
        params['b' + str(i)] = np.zeros((layer_dims[i], 1))

        # Assert to check for correct shapes
        assert(params['W' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert(params['b' + str(i)].shape == (layer_dims[i], 1))

    return params




