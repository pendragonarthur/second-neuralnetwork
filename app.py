import numpy as np

def forward_prop(weights, inputs, biases):
    weighted_sum = np.dot(weights, inputs) + biases
    return weighted_sum

def input_layer(neurons):
    weights = [np.random.uniform(-1.0, 1.0) for _ in range(neurons)]
    bias = [np.random.uniform(-1.0, 1.0) for _ in range(neurons)]
    feedforward = forward_prop(weights, neurons, bias)
    return feedforward

def hidden_layers(inputs, neurons, layers=2):
    for _ in range(layers):
        weights = np.random.uniform(-1.0, 1.0, (neurons, len(inputs)))
        print(weights)
        bias = np.random.uniform(-1.0, 1.0, neurons)
        print(bias)
        layer_output = forward_prop(weights, neurons, bias)
    return layer_output

def output_layer(neurons=10):
    pass