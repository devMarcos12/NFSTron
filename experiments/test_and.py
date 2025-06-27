import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perceptron.neuron import Perceptron
from dataset.logic_problems import get_and_data

def test_and_gate():
    """Test perceptron on AND logic gate"""
    print("=== Testing AND Gate ===")
    
    X, y = get_and_data()
    print(f"Training data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Training data:\n{X}")
    print(f"Labels: {y}")
    
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, activation='step')
    
    print("\nStarting training...")
    perceptron.train(X, y, epochs=100)
    
    print("\nTesting predictions:")
    for i in range(len(X)):
        prediction = perceptron.predict(X[i])
        print(f"Input: {X[i]} | Expected: {y[i]} | Predicted: {prediction}")
    
    # Show final weights and bias
    print(f"\nFinal weights: {perceptron.get_weights()}")
    print(f"Final bias: {perceptron.get_bias()}")
    
    # Show training history
    history = perceptron.get_training_history()
    print(f"Training completed in {len(history['accuracy'])} epochs")
    if history['accuracy']:
        print(f"Final accuracy: {history['accuracy'][-1]:.2f}")

if __name__ == "__main__":
    test_and_gate()