#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of k values to try
k_values = range(1, 7)  # You can adjust the range as needed

best_k = None
best_accuracy = 0.0

for k in k_values:
    # Create a KNN classifier with the current k value and L1-distance (Manhattan) or select (p=1)
    knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Check if this k value gives better accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

    print(f"Accuracy (k={k}): {accuracy}")

print(f"The best k value is {best_k} with accuracy {best_accuracy}")



# In[33]:


# Display some random images and their predicted labels
for idx in range(5):
    random_index = np.random.randint(0, X_test.shape[0])  # Generate a random index within the range
    print(random_index)
    print(X_test.shape)
    image = X_test.iloc[random_index, :].values.reshape(28, 28)
    actual_label = y_test.iloc[random_index]
    predicted_label = y_pred[random_index]

    # Check if the predicted label differs from the actual label
    #if actual_label != predicted_label:
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()
    #else:
    #print("There are no image data with wrong mismatch of the labels")


# In[35]:


# Display some random images and their predicted labels
for idx in range(70):
    random_index = np.random.randint(0, X_test.shape[0])  # Generate a random index within the range
    #print(random_index)
    #print(X_test.shape)
    image = X_test.iloc[random_index, :].values.reshape(28, 28)
    actual_label = y_test.iloc[random_index]
    predicted_label = y_pred[random_index]

    # Check if the predicted label differs from the actual label
    if actual_label != predicted_label:
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}")
        plt.axis('off')
        plt.show()
    #else:
    #print("There are no image data with wrong mismatch of the labels")


# In[ ]:




