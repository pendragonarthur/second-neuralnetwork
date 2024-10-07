import numpy as np

def loss_function(true_pred, model_pred):
    sample = true_pred.shape[1]
    loss = -np.sum(true_pred * np.log(model_pred)) / sample
    return loss

def softmax(x)
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def forward_prop(weights, inputs, biases):
    weighted_sum = np.dot(weights, inputs) + biases
    return weighted_sum

def backward_prop(layers, learning_rate): 
    pass