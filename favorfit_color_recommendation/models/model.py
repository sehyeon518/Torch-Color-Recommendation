import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


def linear(x, weight, bias):
    return np.matmul(x, weight.T) + bias


def relu(x):
    return np.maximum(0, x)


def forward(x, model_weights):
    for key, value in model_weights.items():
        print(f"Layer: {key}, Shape: {np.array(value).shape}")
    # Linear 1
    x = linear(
        x, model_weights["seq_modules.0.weight"], model_weights["seq_modules.0.bias"]
    )
    # LayerNorm 1
    x = layer_norm(
        x,
        gamma=model_weights["seq_modules.1.weight"],
        beta=model_weights["seq_modules.1.bias"],
    )
    x = relu(x)

    # Linear 2
    x = linear(
        x, model_weights["seq_modules.3.weight"], model_weights["seq_modules.3.bias"]
    )
    # LayerNorm 2
    x = layer_norm(
        x,
        gamma=model_weights["seq_modules.4.weight"],
        beta=model_weights["seq_modules.4.bias"],
    )
    x = relu(x)

    # Linear 3
    x = linear(
        x, model_weights["seq_modules.6.weight"], model_weights["seq_modules.6.bias"]
    )
    # ResBlock 1
    for i in range(1, 3):
        weight, bias = (
            model_weights[f"seq_modules.7.layer{i}.weight"],
            model_weights[f"seq_modules.7.layer{i}.bias"],
        )
        x = linear(x, weight, bias)

    # LayerNorm3
    x = layer_norm(
        x,
        gamma=model_weights["seq_modules.8.weight"],
        beta=model_weights["seq_modules.8.bias"],
    )
    x = relu(x)

    # ResBlock 2
    for i in range(1, 3):
        weight, bias = (
            model_weights[f"seq_modules.10.layer{i}.weight"],
            model_weights[f"seq_modules.10.layer{i}.bias"],
        )
        x = linear(x, weight, bias)

    # Output Layer
    x = linear(x, model_weights["out.weight"], model_weights["out.bias"])

    return x
