"""
Perceptron --> Multi Layer Perceptron --> Neural Network --> Deep Neural Network --> Transformer

1. Perceptron
    A perceptron is the simplest type of artificial neuron used in neural networks. It’s essentially a mathematical model that takes multiple inputs, applies weights to them, sums them up, and then passes the result through an activation function to produce an output.

    A perceptron is a binary linear classifier that maps an input vector x to an output using:

    y = f(w.x + b)

    Output = f( (x1 * w1) + (x2 * w2) + (x3 * w3) + bias )

    where w is weight vector, x is input vector, b is bias and f is activation function.

    What is the Activation Function?
        After adding everything up, we get some number like 3.7 or -1.2. We need to convert that to 1 (Win) or 0 (No Win).
        
        We use the simplest possible rule called Step Function:
        
        If sum > 0  →  Output = 1  (WIN ✅)
        If sum ≤ 0  →  Output = 0  (NO WIN ❌)
    
    What is Learning? 🎓
        Learning = adjusting weights automatically until predictions are correct.
        The rule is beautifully simple:

        error = actual_result - predicted_result

        new_weight = old_weight + (learning_rate * error * input)
        
        In plain English:

        If we predicted WIN but team actually LOST → error = -1 → reduce weights
        If we predicted LOSS but team actually WON → error = +1 → increase weights
        If prediction was correct → error = 0 → don't change anything

        Learning rate = how big each adjustment step is (usually small like 0.1)

https://claude.ai/chat/6966c468-0d3c-45d0-83a3-fed1fe06ee68
"""

## Build a Perceptron Neural Network Model from scratch:

class Perceptron:
    def __init__(self, vector, weights, bias=0, learning_rate=0.1):
        self.vector = vector
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
    
    def weighted_sum(self):
        sum = 0

        for v, w in (self.vector, self.weights):
            sum += v * w

        return sum + self.bias
    
    def activation_fn(self, weighted_sum):
        return 1 if weighted_sum > 0 else 0
    
    def calculate_error(self, actual, prediction):
        return actual - prediction
    
    def update_weight(self, old_weight, error, input):
        return old_weight + (self.learning_rate * error * input)
    
    def train(self, data):
        pass


import pandas as pd
import numpy as np

def match_win_prediction():
    # Prepare dataset read csv, drop first row, convert to pandas dataframe
    data = pd.read_csv("match_win_prediction.csv")
    data = data.drop(0)
    data = data.reset_index(drop=True)
    # print(data[:5])

    # convert to numpy array
    data = data.to_numpy()
    # print(data[:5])

    # separate features and target
    features = data[:, :-1]
    target = data[:, -1]
    print("Features:", features.shape)
    print("Target:", target.shape)
    print(features[:5])
    print(target[:5])

    # initialize weights as 0
    weights = np.zeros(features.shape)
    print("Weights:", weights.shape)
    print(weights[:2])


match_win_prediction()
    