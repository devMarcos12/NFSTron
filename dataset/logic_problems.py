import numpy as np

def get_and_data():
    """
    Returns training data for the AND logic gate.
    
    Returns:
        tuple: (X, y) where X is input data and y is labels
    """
    X = np.array([
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([0, 0, 0, 1])
    
    print(f"DEBUG - X shape in get_and_data: {X.shape}")
    print(f"DEBUG - X content: {X}")
    
    return X, y

def get_or_data():
    """
    Returns training data for the OR logic gate.
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0], 
        [1, 1]
    ])
    
    y = np.array([0, 1, 1, 1])
    
    return X, y

def get_xor_data():
    """
    Returns training data for the XOR logic gate.
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([0, 1, 1, 0])
    
    return X, y