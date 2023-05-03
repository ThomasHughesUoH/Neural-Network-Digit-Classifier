import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the dataset and split it into training and testing sets
digits = load_digits()

# Save the dataset to a file
np.savetxt("digits_data.txt", digits.data)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Define and train the classifier, tracking accuracy and loss over time
clf = MLPClassifier(hidden_layer_sizes=(100, 75), max_iter=1000, alpha=8e-6, solver='adam', verbose=10, random_state=22, tol=1e-9)
training_accuracy, training_loss, testing_accuracy, testing_loss = [], [], [], []
for i in range(1000):
    clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
    training_accuracy.append(clf.score(X_train, y_train))
    training_loss.append(log_loss(y_train, clf.predict_proba(X_train)))
    testing_accuracy.append(clf.score(X_test, y_test))
    testing_loss.append(log_loss(y_test, clf.predict_proba(X_test)))

# Use the trained classifier to predict on the testing set and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Plot accuracy and loss over time
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
for ax, metric, name in zip(axs, [training_accuracy, testing_accuracy], ['Training Accuracy', 'Testing Accuracy']):
    ax.plot(metric, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(name)
    ax.legend()
for ax, metric, name in zip(axs, [training_loss, testing_loss], ['Training Loss', 'Testing Loss']):
    ax.plot(metric, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(name)
    ax.legend()
plt.savefig('loss_and_accuracy.svg', dpi=300)
plt.show()

# Plot some sample images from the dataset with their predicted labels
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
for ax, idx in zip(axs.ravel(), range(16)):
    ax.imshow(X_test[idx].reshape((8, 8)), cmap=plt.cm.gray_r)
    ax.set_title(f"Predicted: {y_pred[idx]}\nActual: {y_test[idx]}")
    ax.set_xticks(())
    ax.set_yticks(())
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('sample_images.svg', dpi=300)
plt.show()

# Print the accuracy of the classifier
print("\nAccuracy:", accuracy,'\n')

with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Predicted", "Actual"])
    for i in range(len(y_pred)):
        writer.writerow([y_pred[i], y_test[i]])
    print("Data written to predictions.csv\n")

with open("predictions.csv", "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    incorrect_predictions = []
    for row in reader:
        predicted, actual = map(int, row)
        if predicted != actual:
            incorrect_predictions.append((predicted, actual))

    if incorrect_predictions:
        print("Incorrect Predictions:\n", "==================")
        for prediction in incorrect_predictions:
            print(f"Predicted: {prediction[0]} Actual: {prediction[1]}")
        
        # Create a dictionary to count the number of occurrences of each number in incorrect_predictions
        error_counts = {}
        for prediction in incorrect_predictions:
            actual = prediction[1]
            if actual in error_counts:
                error_counts[actual] += 1
            else:
                error_counts[actual] = 1
        
        # Create a bar chart of the error counts
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(error_counts.keys(), error_counts.values())
        ax.set_xlabel('Number')
        ax.set_ylabel('Frequency of Errors')
        ax.set_title('Frequency of Errors in Predictions')
        plt.show()
        
    else:
        print("All Predictions are Correct")
