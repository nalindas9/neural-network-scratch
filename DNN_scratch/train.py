from utils import *

def main():
    layer_dims = [5,2,3,5,7]
    params = initialize_parameters(layer_dims)
    print('Params: ', params['b3'].shape)

if __name__ == "__main__":
    main()