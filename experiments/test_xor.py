import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perceptron.mlp import MLP
from dataset.logic_problems import get_xor_data

def test_mlp_xor():
    """Test MLP on XOR problem"""
    print("=== Testing MLP on XOR Gate ===")
    print("This should work, unlike the simple perceptron!")
    
    # Get data
    X, y = get_xor_data()
    print(f"Training data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Create MLP: 2 inputs -> 4 hidden -> 1 output
    mlp = MLP(layers=[2, 4, 1], learning_rate=0.5, activation='sigmoid')
    
    print("\nStarting training...")
    mlp.train(X, y, epochs=2000, verbose=True)
    
    # Test predictions
    print("\nTesting predictions:")
    predictions = mlp.predict_binary(X)
    
    for i in range(len(X)):
        prob = mlp.predict(X[i])[0]
        print(f"Input: {X[i]} | Expected: {y[i]} | Predicted: {predictions[i]} ({prob:.3f})")
    
    # Final accuracy
    accuracy = np.mean(predictions == y)
    print(f"\nFinal accuracy: {accuracy:.2f}")
    
    # Show training history
    history = mlp.get_training_history()
    print(f"Training completed in {len(history['loss'])} epochs")
    print(f"Final loss: {history['loss'][-1]:.4f}")

if __name__ == "__main__":
    import numpy as np
    test_mlp_xor()