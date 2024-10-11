import numpy as np

def loss_function(true_pred, model_pred):
    m = true_pred.shape[1] # m = numero de amostrar no conjunto de dados
    loss = -np.sum(true_pred * np.log(model_pred)) / m
    return loss

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def relu(x):
    return np.maximum(0, x)

def initialize_weights(neurons, inputs): # inicialização HE
    dev = np.sqrt(2 / inputs) # desvio padrão
    weights = np.random.randn(neurons, inputs) * dev 
    return weights

def initialize_biases(neurons):
    bias_matrix = np.zeros((neurons, 1)) # cria uma matriz de 0 de 'neurons' linhas e 1 coluna
    return bias_matrix


def forward_prop(weights, inputs, biases):
    weighted_sum = relu(np.dot(weights, inputs) + biases)
    return weighted_sum

def backward_prop(final_output, hidden_output, true_pred, learning_rate, weights, biases, inputs):
    error_output = final_output - true_pred

    error_hidden = np.dot(weights['output'].T, error_output) * (hidden_output * (1 - hidden_output))

    weights['output'] -= learning_rate * np.dot(error_output, hidden_output.T)
    biases['output'] -= learning_rate * np.sum(error_output, axis=1, keepdims=True)

    weights['hidden'] -= learning_rate * np.dot(error_hidden, inputs.T)
    biases['hidden'] -= learning_rate * np.sum(error_hidden, axis=1, keepdims=True)

    return weights, biases 

    