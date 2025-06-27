import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perceptron.neuron import Perceptron
from dataset.logic_problems import get_or_data

def test_or_gate():
    """Test perceptron on OR logic gate"""
    print("=== Testing OR Gate ===")
    
    X, y = get_or_data()
    print(f"Training data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, activation='step')
    perceptron.train(X, y, epochs=100)
    
    print("\nTesting predictions:")
    for i in range(len(X)):
        prediction = perceptron.predict(X[i])
        print(f"Input: {X[i]} | Expected: {y[i]} | Predicted: {prediction}")
    
    print(f"\nFinal weights: {perceptron.get_weights()}")
    print(f"Final bias: {perceptron.get_bias()}")
    
    history = perceptron.get_training_history()
    print(f"Training completed in {len(history['accuracy'])} epochs")
    if history['accuracy']:
        print(f"Final accuracy: {history['accuracy'][-1]:.2f}")

if __name__ == "__main__":
    test_or_gate()