import numpy as np


def linear(input, weight, bias):
    return np.matmul(input, weight.T) + bias

def relu(x):
    return np.maximum(0, x)

def res_block(input, block_weights, block_weights_norm, block_biases_norm):
    residue = input
    x = linear(input, block_weights[0], block_biases_norm[0])
    x = relu(x)
    x = linear(x, block_weights[1], block_biases_norm[1])
    return x + residue

def forward(x, model_weights):

    # Linear 1
    x = linear(x, model_weights['seq_modules.0.weight'], model_weights['seq_modules.0.bias'])
    # LayerNorm 1    
    x = x * model_weights['seq_modules.1.weight'] + model_weights['seq_modules.1.bias']
    x = relu(x)

    # Linear 2
    x = linear(x, model_weights['seq_modules.3.weight'], model_weights['seq_modules.3.bias'])
    # LayerNorm 2
    x = x * model_weights['seq_modules.4.weight'] + model_weights['seq_modules.4.bias']
    x = relu(x)

    # Linear 3
    x = linear(x, model_weights['seq_modules.6.weight'], model_weights['seq_modules.6.bias'])
    # ResBlock 1
    for i in range(1, 3):
        weight, bias = model_weights[f'seq_modules.7.layer{i}.weight'], model_weights[f'seq_modules.7.layer{i}.bias']
        x = linear(x, weight, bias)

    # Linear 4
    x = x * model_weights['seq_modules.8.weight'] + model_weights['seq_modules.8.bias']
    x = relu(x)

    # ResBlock 2
    for i in range(1, 3):
        weight, bias = model_weights[f'seq_modules.10.layer{i}.weight'], model_weights[f'seq_modules.10.layer{i}.bias']
        x = linear(x, weight, bias)

    # Output Layer
    x = linear(x, model_weights['out.weight'], model_weights['out.bias'])

    return x
