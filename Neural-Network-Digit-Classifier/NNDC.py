# Import necessary libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the dataset
digits = load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Define the neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(100,75), max_iter=1000, alpha=0.000008, solver='adam', verbose=10,  random_state=22, tol=0.000000001)

# Train the classifier on the training set
training_accuracy = []
training_loss = []
testing_accuracy = []
testing_loss = []

for i in range(1000):
    clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
    training_accuracy.append(clf.score(X_train, y_train))
    training_loss.append(log_loss(y_train, clf.predict_proba(X_train)))
    testing_accuracy.append(clf.score(X_test, y_test))
    testing_loss.append(log_loss(y_test, clf.predict_proba(X_test)))

# Use the trained classifier to predict on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Plot the accuracy and loss over time
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax[0].plot(training_accuracy, label="Training Accuracy")
ax[0].plot(testing_accuracy, label="Testing Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Accuracy Over Time")
ax[0].legend()

ax[1].plot(training_loss, label="Training Loss")
ax[1].plot(testing_loss, label="Testing Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_title("Loss Over Time")
ax[1].legend()

plt.savefig('loss_and_accuracy.svg', dpi=300)
plt.show()

# Plot some sample images from the dataset with their predicted labels
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
for i in range(16):
    row = i // 4
    col = i % 4
    ax[row, col].imshow(X_test[i].reshape((8, 8)), cmap=plt.cm.gray_r)
    ax[row, col].set_title("Predicted: {}\nActual: {}".format(y_pred[i], y_test[i]))
    ax[row, col].set_xticks(())
    ax[row, col].set_yticks(())

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('sample_images.svg', dpi=300)
plt.show()

# Print the accuracy of the classifier
print("Accuracy:", accuracy)
