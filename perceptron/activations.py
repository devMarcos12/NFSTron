import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh function"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return (x > 0).astype(float)

def step(x):
    """Step activation function"""
    return (x >= 0).astype(int)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative),
    'step': (step, None)
}

def get_activation(name):
    """Get activation function and its derivative"""
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation '{name}' not supported. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]