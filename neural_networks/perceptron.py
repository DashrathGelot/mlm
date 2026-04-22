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
import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, number_of_features, learning_rate=0.1):
        self.weights = np.zeros(number_of_features)
        self.bias = 0.0
        self.learning_rate = learning_rate
    
    def weighted_sum(self, input_vector):
        total = 0
        for v, w in zip(input_vector, self.weights):
            total += v * w
        return total + self.bias
    
    def activation_fn(self, weighted_sum):
        return 1 if weighted_sum > 0 else 0
    
    def calculate_error(self, actual, prediction):
        return actual - prediction
    
    def update_weight(self, old_weight, error, input_feature_value):
        return old_weight + (self.learning_rate * error * input_feature_value)
    
    def update_weights(self, error, input):
        for col in range(len(input)):
            self.weights[col] = self.update_weight(self.weights[col], error, input[col])
        
        self.bias = self.update_weight(self.bias, error, 1)
    
    def epoch_train(self, train_features, train_target):
        wrong_predictions = 0

        for i in range(len(train_features)):    
            w_sum = self.weighted_sum(train_features[i])

            prediction = self.activation_fn(w_sum)
            
            err = self.calculate_error(train_target[i], prediction)
            
            self.update_weights(err, train_features[i])
           
            if err != 0:
                wrong_predictions += 1
                
        return wrong_predictions

    def train(self, train_features, train_target, epochs=1000):
        for epoch in range(epochs):
            print("Epoch : ", epoch)
            wrong_predictions = self.epoch_train(train_features, train_target)
            print("Wrong Predictions: ", wrong_predictions)
            if wrong_predictions == 0:
                print("Model converged")
                break
        print("Final Weights: ", self.weights)
        print("Final Bias ", self.bias)
    
    def test(self, test_features, test_target):
        correct_predictions = 0
        for i in range(len(test_features)):
            w_sum = self.weighted_sum(test_features[i])
            prediction = self.activation_fn(w_sum)
            if prediction == test_target[i]:
                correct_predictions += 1
        print("Correct Predictions: ", correct_predictions)
        accuracy = (correct_predictions / len(test_features)) * 100
        print(f"\n🎯 Test Results: {correct_predictions}/{len(test_features)} correct")
        print(f"📊 Accuracy: {accuracy:.1f}%")
        return correct_predictions


def get_data():
    # Prepare dataset read csv, drop first row, convert to pandas dataframe
    data = pd.read_csv("match_win_prediction.csv")
    # print(data[:5])

    # convert to numpy array
    data = data.to_numpy()
    # print(data[:5])

    # separate features and target
    features = data[:, :-1].astype(float)
    target = data[:, -1].astype(int) 
    print(f"Dataset: {features.shape[0]} matches, {features.shape[1]} features\n")
    return features, target

def train_test_pipeline(features, target):
    # train and test split
    train_test_split = 0.8
    train_size = int(len(features) * train_test_split)

    train_features = features[:train_size]
    train_target = target[:train_size]
    test_features = features[train_size:]
    test_target = target[train_size:]

    print("Train Features:", train_features.shape)
    print("Train Target:", train_target.shape)
    print("Test Features:", test_features.shape)
    print("Test Target:", test_target.shape)

    # train the model
    perceptron = Perceptron(number_of_features=train_features.shape[1])
    perceptron.train(train_features, train_target)
    perceptron.test(test_features, test_target)

def match_win_prediction():
    features, target = get_data()
    train_test_pipeline(features, target)

match_win_prediction()

"""
Output of above code:

Final Weights:  [ -4.2    1.4    4.9   -0.33 -20.31]
Final Bias  -5.099999999999998
Correct Predictions:  10

🎯 Test Results: 10/10 correct
📊 Accuracy: 100.0%

if you see weights are negative it means the feature is not important for the model
and here is our features: form,opponent_strength,home,goals_scored,goals_conceded
now in football world it doesn't make any sense because 
model learned -> "Better form → LESS likely to win?"
"Stronger opponent → MORE likely to win?"
"More goals scored → LESS likely to win?"
-- "I only care about goals conceded. Everything else barely matters."

## The Core Problem: Feature Scaling

Here's the visual of what's happening:
Feature          Range        Weight
form             8-14    →   -4.2    (confused by large range)
opponent_str     40-80   →    1.4    (confused by huge range)
home             0-1     →    4.9    ✅ (small range, learned correctly)
goals_scored     1.5-3.0 →   -0.33  (confused)
goals_conceded   0.8-2.0 →  -20.31  (overcompensating for small range)

The model is fighting against different scales instead of learning football patterns.
The fix is called Normalization — scaling ALL features to the same range (0 to 1):
normalized = (value - min) / (max - min)

form 12 → (12-8)/(14-8) = 0.67
opp  70 → (70-40)/(80-40) = 0.75
home  1 → (1-0)/(1-0) = 1.0

Now ALL features are between 0 and 1. Model can fairly compare them.

now let's add normalization to our model to make all scale to 0 to 1
"""

def normalize(features):
    min_vals = features.min(axis=0)   # min of each feature
    max_vals = features.max(axis=0)   # max of each feature
    return (features - min_vals) / (max_vals - min_vals)

def match_win_prediction_with_normalization():
    features, target = get_data()
    # Before training:
    features = normalize(features)
    train_test_pipeline(features, target)

match_win_prediction_with_normalization()

"""
Now it does make sense but still because of less data model learn one thing wrong

📊 Weight Ranking — What Model Thinks Matters Most
Feature             Weight    Importance    Makes Sense?
─────────────────────────────────────────────────────
goals_conceded      -0.125    🥇 Highest    ✅ Yes
home                 0.200    🥈 Second     ✅ Yes
goals_scored         0.042    🥉 Third      ✅ Yes
opponent_strength    0.041    4th           ❌ Wrong direction
form                 0.031    5th           ✅ Yes
bias                -0.100    default lean  (leans toward no-win by default)
4 out of 5 features learned correctly. Pretty good for 50 samples! 🎯

🎯 Bias = -0.1 — What Does This Mean?
bias = -0.100  (negative)
This means:

"When all features are average (0.5 after normalization) — model leans slightly toward NO WIN by default"

In football terms:

"When I know nothing special about a team — I assume they probably won't win"

Actually makes sense — in any match, only ONE team wins. So default lean toward no-win is logical.


💡 The Big Picture Lesson
Before normalization:
weights = [-4.2, 1.4, 4.9, -0.33, -20.31]
→ Nonsense. Model fighting scales. One feature dominating.

After normalization:
weights = [0.031, 0.041, 0.200, 0.042, -0.125]
→ Meaningful. Balanced. Interpretable. Makes football sense.
Same accuracy. Completely different understanding.
This is why normalization is not optional in real ML — it's mandatory.
"""