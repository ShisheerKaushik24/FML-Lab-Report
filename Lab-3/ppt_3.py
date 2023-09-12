import numpy as np
import matplotlib.pyplot as plt

# Create and train the perceptron
perceptron = Perceptron(num_features=3)
perceptron.train(training_data, labels)

# Generate points for plotting the decision boundary
x_vals = np.linspace(0, 300, 100)
y_vals = -(perceptron.weights[1] * x_vals + perceptron.weights[0]) / perceptron.weights[2]

# Scatter plot of training data
plt.scatter(training_data[labels == 1][:, 0], training_data[labels == 1][:, 1], color='red', label='Apple')
plt.scatter(training_data[labels == -1][:, 0], training_data[labels == -1][:, 1], color='orange', label='Orange')

# Plot the decision boundary
plt.plot(x_vals, y_vals, label='Decision Boundary')

plt.xlabel('Weight')
plt.ylabel('Redness')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.show()
