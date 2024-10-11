import numpy as np
from functions import forward_prop, softmax, initialize_biases, initialize_weights

def input_layer(neurons):
    pass

def hidden_layers(inputs, neurons, layers=2):
    weights_dict = []
    biases_dict = []

    for _ in range(layers):
        weights = initialize_weights(neurons, inputs)
        bias = initialize_biases(neurons)
        layer_output = forward_prop(weights, neurons, bias)

        weights_dict.append(weights)
        biases_dict.append(bias)        
    return layer_output, weights_dict, biases_dict

def output_layer(inputs, neurons=10):
    weights = np.random.uniform(-1.0, 1.0, (neurons, len(inputs)))
    bias = np.random.uniform(-1.0, 1.0, neurons)
    logits = forward_prop(weights, neurons, bias)
    final_output = softmax(logits)
    return final_output, weights, bias