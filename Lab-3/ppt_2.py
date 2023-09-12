import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.1, epochs=1000):
        self.weights = np.random.rand(num_features + 1)  # +1 for the bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activate(summation)

    def train(self, training_data, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                update = self.learning_rate * (label - prediction)
                self.weights[1:] += update * inputs
                self.weights[0] += update
                print('updated')

# Labels: 1 for Apple, -1 for Orange
labels = np.array([1, 1, 1, 1,-1,-1,-1,-1,-1])

# Create and train the perceptron
perceptron = Perceptron(num_features=3)
perceptron.train(training_data, labels)

# Test the trained perceptron
test_data = np.array([
    [245,130,90],  # Apple
    [50,30,40],  # Orange
    [222,110,70],   # Apple
    [45,90,30],  # Orange
])

for data in test_data:
    prediction = perceptron.predict(data)
    if prediction == 1:
        print("Apple")
    else:
        print("Orange")
