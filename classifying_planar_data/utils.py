import numpy as np

# Function to compute sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x)) 
