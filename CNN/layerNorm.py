import numpy as np

class layerNorm:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.mean = None
        self.variance = None

    def forward(self, x):
        # Compute mean and variance along the feature dimension
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.variance = np.var(x, axis=1, keepdims=True)

        # Normalize the input
        normalized_x = (x - self.mean) / np.sqrt(self.variance + self.epsilon)

        return normalized_x
    
    def backward(self, dL_dy):
        return dL_dy * (1 / np.sqrt(self.variance + self.epsilon))