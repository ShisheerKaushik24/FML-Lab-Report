#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample synthetic dataset, 1=Apple, -1=Orange
# Generate 40 additional random data points (20 per class)

def get_data(n):
    rand_samples = n
    samples = []

    # Generating random samples for class 1 (Apple)
    for _ in range(rand_samples // 2):
        redness = np.random.randint(200, 255)
        weight = np.random.randint(100, 140)
        samples.append([redness, weight, 1])

    # Generating random samples for class -1 (Orange)
    for _ in range(rand_samples // 2):
        redness = np.random.randint(10, 50)
        weight = np.random.randint(80, 100)
        samples.append([redness, weight, -1])

    return np.array(samples)


# In[9]:


data = get_data(50)
print(data)


# In[10]:


data.shape


# In[11]:


# Separate features (redness and weight) and labels (fruit type)
X = data[:, :2]
y = data[:, 2]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a KNN classifier with k=1
knn_classifier = KNeighborsClassifier(n_neighbors=2)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

print(X_test)
# Predict the fruit types for the test data
y_pred = knn_classifier.predict(X_test)
print(y_pred)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Convert the data to a pandas DataFrame for visualization
df = pd.DataFrame(data, columns=['Redness', 'Weight', 'Fruit Type'])

# Plot the data samples
plt.figure(figsize=(8, 6))
for fruit_type in df['Fruit Type'].unique():
    subset = df[df['Fruit Type'] == fruit_type]
    plt.scatter(subset['Redness'], subset['Weight'], label=fruit_type)

plt.xlabel('Redness Value')
plt.ylabel('Weight (grams)')
plt.title('Apple vs Orange Classification')
plt.legend()
plt.show()


# In[13]:


data_1 = get_data(50)
print(data_1)


# In[14]:


# Separate features (redness and weight) and labels (fruit type)
X = data_1[:, :2]
y = data_1[:, 2]

# Split dataset into training (80%), testing (10%), and validation (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a KNN classifier with k=1
knn_classifier = KNeighborsClassifier(n_neighbors=2)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

print(X_test)
# Predict the fruit types for the test data
y_pred = knn_classifier.predict(X_test)
print(y_pred)

# Calculate the accuracy of the model on the validation set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Convert the data to a pandas DataFrame for visualization
df = pd.DataFrame(data_1, columns=['Redness', 'Weight', 'Fruit Type'])

# Plot the data samples
plt.figure(figsize=(8, 6))
for fruit_type in df['Fruit Type'].unique():
    subset = df[df['Fruit Type'] == fruit_type]
    plt.scatter(subset['Redness'], subset['Weight'], label=fruit_type)

plt.xlabel('Redness Value')
plt.ylabel('Weight (grams)')
plt.title('Apple vs Orange Classification')
plt.legend()
plt.show()


# In[16]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample synthetic dataset, 1=Apple, -1=Orange
# Generate 40 additional random data points (20 per class)

def get_data(n):
    rand_samples = n
    samples = []

    # Generating random samples for class 1 (Apple)
    for _ in range(rand_samples // 2):
        redness = np.random.randint(200, 255)
        weight = np.random.randint(100, 140)
        samples.append([redness, weight, 1])

    # Generating random samples for class -1 (Orange)
    for _ in range(rand_samples // 2):
        redness = np.random.randint(10, 50)
        weight = np.random.randint(80, 100)
        samples.append([redness, weight, -1])

    return np.array(samples)

data_2 = get_data(50)
print(data_2)

# Separate features (redness and weight) and labels (fruit type)
X = data_2[:, :2]
y = data_2[:, 2]

# Split dataset into training (80%), testing (10%), and validation (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Values of k to try
k_values = [3, 5, 7]

for k in k_values:
    # Create a KNN classifier with the current k value
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Predict the fruit types for the validation data
    y_pred_val = knn_classifier.predict(X_val)

    # Calculate the accuracy of the model on the validation set
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy (k={k}):", accuracy_val)

    # Predict the fruit types for the test data
    y_pred_test = knn_classifier.predict(X_test)

    # Calculate the accuracy of the model on the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy (k={k}):", accuracy_test)
    print()


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Sample synthetic dataset, 1=Apple, -1=Orange
# Generate 40 additional random data points (20 per class)

def get_data(n):
    rand_samples = n
    samples = []

    # Generating random samples for class 1 (Apple)
    for _ in range(rand_samples // 2):
        redness = np.random.randint(200, 255)
        weight = np.random.randint(100, 140)
        samples.append([redness, weight, 1])

    # Generating random samples for class -1 (Orange)
    for _ in range(rand_samples // 2):
        redness = np.random.randint(10, 50)
        weight = np.random.randint(80, 100)
        samples.append([redness, weight, -1])

    return np.array(samples)

data_3 = get_data(50)
print(data_3)

# Separate features (redness and weight) and labels (fruit type)
X = data_3[:, :2]
y = data_3[:, 2]

# Split dataset into training (80%), testing (10%), and validation (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Values of k to try
k_values = [3, 5, 7]

for k in k_values:
    # Create a KNN classifier with the current k value
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Predict the fruit types for the validation data
    y_pred_val = knn_classifier.predict(X_val)

    # Calculate the accuracy of the model on the validation set
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy (k={k}):", accuracy_val)

    # Calculate the confusion matrix for validation set
    cm_val = confusion_matrix(y_val, y_pred_val)
    print("Confusion Matrix (Validation):")
    print(cm_val)
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Validation) - k={k}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    print()

