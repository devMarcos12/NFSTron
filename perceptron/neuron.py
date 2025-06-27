import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.01, activation='step'):
        self.n_inputs = n_inputs
        self.lr = learning_rate
        self.weights = np.zeros(n_inputs) 
        self.bias = 0.0
        self.activation = activation
        self.training_history = {'erros': [], 'accuracy': []}

        if n_inputs <= 0:
            raise ValueError('Number of inputs must be a positive integer.')
        if activation not in ['step', 'sigmoid', 'tanh', 'relu']:
            raise ValueError(f'Unsupported activation function: {activation}. Supported: step, sigmoid, tanh, relu.')
        if learning_rate <= 0:
            raise ValueError('Learning rate must be a positive number.')

    def _activation_function(self, x):
        """Apply the activation function."""

        if hasattr(x, 'item'):
            x = x.item()
        
        if self.activation == 'step':
            return 1 if x >= 0 else 0
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return max(0, x)
        else:
            raise ValueError(f'Unsupported activation function: {self.activation}')

    def predict(self, X):
        """Make predictions using the trained perceptron."""
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation_function(linear_output)
            
    def train(self, training_data, labels, epochs):
        """Train the perceptron using the provided training data and labels."""

        training_data = np.array(training_data)
        labels = np.array(labels)

        for epoch in range(epochs):
            errors = 0
            correct_predictions = 0

            for i in range(len(training_data)):
                prediction = self.predict(training_data[i])
                
                expected = int(labels[i])
                predicted = int(prediction)
                error = expected - predicted

                if error != 0:
                    errors += 1
                    self.weights += self.lr * error * training_data[i]
                    self.bias += self.lr * error
                else:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(training_data)

            self.training_history['erros'].append(errors)
            self.training_history['accuracy'].append(accuracy)

            if errors == 0:
                print(f'Training completed in {epoch + 1} epochs with no errors.')
                break

    def get_weights(self):
        """Return the current weights of the perceptron."""
        return self.weights.copy()
    
    def get_bias(self):
        """Return the current bias of the perceptron."""
        return self.bias

    def get_training_history(self):
        """Return the training history of the perceptron."""
        return self.training_history.copy()