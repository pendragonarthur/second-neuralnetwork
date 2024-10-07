import numpy as np
from functions import forward_prop

def input_layer(neurons):
    pass

def hidden_layers(inputs, neurons, layers=2):
    for _ in range(layers):
        weights = np.random.uniform(-1.0, 1.0, (neurons, len(inputs)))
        print(weights)
        bias = np.random.uniform(-1.0, 1.0, neurons)
        print(bias)
        layer_output = forward_prop(weights, neurons, bias)
    return layer_output

def output_layer(inputs, neurons=10):
    weights = np.random.uniform(-1.0, 1.0, (neurons, len(inputs)))
    bias = np.random.uniform(-1.0, 1.0, neurons)
    final_output = forward_prop(weights, neurons, bias)
    return final_output


    