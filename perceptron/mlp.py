import numpy as np
from .activations import get_activation

class MLP:
    def __init__(self, layers, learning_rate=0.01, activation='sigmoid'):
        """
        Multi-Layer Perceptron (MLP) constructor.

        Args:
            layers (list): List of integers representing the number of neurons in each layer.
            learning_rate (float): Learning rate for weight updates.
            activation (str): Activation function for hidden layers. 
        """

        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.activation_func, self.activation_derivative = get_activation(activation)

        if len(layers) < 2:
            raise ValueError("MLP must have at least two layers (input and output).")
        if any(l <= 0 for l in layers):
            raise ValueError("All layers must have a positive number of neurons.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            bias = np.zeros((1, layers[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)

        self.training_history = {'loss': [], 'accuracy': []}

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            List of activations for each layer.
        """
        activations = [X]

        for i in range(len(self.weights)):
            # Calculate the linear combination
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation_func(z)
            activations.append(a)

        return activations

    def backward(self, X, y, activations):
        """
        Backward pass through the network to compute gradients.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples, n_outputs).
            activations (list): List of activations from the forward pass.
        """
        m = X.shape[0]
        output_error = activations[-1] - y.reshape(-1, 1)

        weight_gradients = []
        bias_gradients = []

        error = output_error

        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, error) / m
            dB = np.sum(error, axis=0, keepdims=True) / m

            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, dB)

            # Calculate the error for the previous layer
            if i > 0:
                error = np.dot(error, self.weights[i].T)
                z = np.dot(activations[i-1], self.weights[i-1]) + self.biases[i-1]
                error = error * self.activation_derivative(z)

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def train(self, X, y, epochs:int = 1000, verbose: bool =True):
        """
        Train the neural network.

        Args:
            X: Training data
            y: Labels
            epochs: Number of epochs
            verbose: Show progress
        """
        X = np.array(X)
        y = np.array(y)

        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)
            predictions = activations[-1]

            # Calculate loss (Mean Squared Error)
            loss = np.mean((predictions.flatten() - y)**2)

            # Calculate accuracy (for binary classification)
            pred_binary = (predictions.flatten() > 0.5).astype(int)
            accuracy = np.mean(pred_binary == y)

            # Save history
            self.training_history['loss'].append(loss)
            self.training_history['accuracy'].append(accuracy)

            # Backward pass
            self.backward(X, y, activations)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

            # Early stopping if converged
            if accuracy == 1.0 and loss < 0.01:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        activations = self.forward(X)
        return activations[-1].flatten()

    def predict_binary(self, X):
        """Binary predictions (0 or 1)"""
        predictions = self.predict(X)
        return (predictions > 0.5).astype(int)

    def get_training_history(self):
        """Return training history"""
        return self.training_history.copy()