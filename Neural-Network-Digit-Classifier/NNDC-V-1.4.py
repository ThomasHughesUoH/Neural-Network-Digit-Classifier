import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

class DigitClassifier:
    def __init__(self):
        """
        Initialize the DigitClassifier object.
        """
        self.clf = None
        self.training_accuracy = []
        self.training_loss = []
        self.testing_accuracy = []
        self.testing_loss = []
        self.y_pred = []
        self.accuracy = 0.0
        self.incorrect_predictions = []
        self.error_counts = {}

    def load_digits_dataset(self):
        """
        Load the digits dataset and split it into training and testing sets.

        Returns:
            X_train (numpy.ndarray): Training data features.
            X_test (numpy.ndarray): Testing data features.
            y_train (numpy.ndarray): Training data labels.
            y_test (numpy.ndarray): Testing data labels.
        """
        digits = load_digits()
        np.savetxt("digits_data.txt", digits.data)
        X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_classifier(self, X_train, X_test, y_train, y_test):
        """
        Train the neural network classifier.

        Args:
            X_train (numpy.ndarray): Training data features.
            X_test (numpy.ndarray): Testing data features.
            y_train (numpy.ndarray): Training data labels.
            y_test (numpy.ndarray): Testing data labels.
        """
        self.clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, alpha=8e-6, solver='adam',
                                 verbose=30, random_state=22, tol=1e-9)
        for i in range(2000):
            self.clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
            self.training_accuracy.append(self.clf.score(X_train, y_train))
            self.training_loss.append(log_loss(y_train, self.clf.predict_proba(X_train)))
            self.testing_accuracy.append(self.clf.score(X_test, y_test))
            self.testing_loss.append(log_loss(y_test, self.clf.predict_proba(X_test)))

    def predict(self, X_test, y_test):
        """
        Perform predictions on the testing data.

        Args:
            X_test (numpy.ndarray): Testing data features.
            y_test (numpy.ndarray): Testing data labels.
        """
        self.y_pred = self.clf.predict(X_test)
        self.accuracy = accuracy_score(y_test, self.y_pred)

    def collect_incorrect_predictions(self, predictions):
        """
        Collect the incorrect predictions from the given list of predictions.

        Args:
            predictions (list): List of predicted and actual labels.

        Returns:
            None
        """
        self.incorrect_predictions = []
        for prediction in predictions:
            predicted, actual = map(int, prediction)
            if predicted != actual:
                self.incorrect_predictions.append((predicted, actual))

    def count_error_occurrences(self):
        """
        Count the occurrences of each error label in the incorrect predictions.

        Returns:
            None
        """
        self.error_counts = {}
        for prediction in self.incorrect_predictions:
            actual = prediction[1]
            if actual in self.error_counts:
                self.error_counts[actual] += 1
            else:
                self.error_counts[actual] = 1

    def plot_loss_and_accuracy(self):
        """
        Plot the training and testing loss and accuracy over epochs.

        Returns:
            None
        """
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
        for ax, metric, name in zip(axs, [self.training_accuracy, self.testing_accuracy],
                                    ['Training Accuracy', 'Testing Accuracy']):
            ax.plot(metric, label=name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(name)
            ax.legend()
        for ax, metric, name in zip(axs, [self.training_loss, self.testing_loss], ['Training Loss', 'Testing Loss']):
            ax.plot(metric, label=name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(name)
            ax.legend()
        plt.savefig('loss_and_accuracy.svg', dpi=300)
        plt.show()

    def plot_sample_images(self, X_test, y_pred, y_test):
        """
        Plot a grid of sample images along with their predicted and actual labels.

        Args:
            X_test (numpy.ndarray): Testing data features.
            y_pred (numpy.ndarray): Predicted labels.
            y_test (numpy.ndarray): Actual labels.

        Returns:
            None
        """
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
        for ax, idx in zip(axs.ravel(), range(16)):
            ax.imshow(X_test[idx].reshape((8, 8)), cmap=plt.cm.gray_r)
            ax.set_title(f"Predicted: {y_pred[idx]}\nActual: {y_test[idx]}")
            ax.set_xticks(())
            ax.set_yticks(())
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('sample_images.svg', dpi=300)
        plt.show()

    def save_predictions_to_csv(self, filename, y_test):
        """
        Save the predicted and actual labels to a CSV file.

        Args:
            filename (str): Name of the CSV file.
            y_test (numpy.ndarray): Actual labels.

        Returns:
            None
        """
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Predicted", "Actual", "Match"])
            for i in range(len(self.y_pred)):
                predicted = self.y_pred[i]
                actual = y_test[i]
                match = int(predicted == actual)
                writer.writerow([predicted, actual, match])
            print("Data written to predictions.csv\n")

    def load_predictions_from_csv(self, filename):
        """
        Load the predicted and actual labels from a CSV file.

        Args:
            filename (str): Name of the CSV file.

        Returns:
            predictions (list): List of predicted and actual labels.
        """
        with open(filename, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            predictions = []
            for row in reader:
                predicted, actual, _ = map(int, row)
                predictions.append((predicted, actual))
        return predictions

    def learn_to_draw(self, X_correct, y_correct):
        """
        Train the neural network classifier to learn to draw digits.

        Args:
            X_correct (numpy.ndarray): Correctly identified digit features.
            y_correct (numpy.ndarray): Correctly identified digit labels.

        Returns:
            None
        """
        self.clf = MLPClassifier(hidden_layer_sizes=(400, 300), max_iter=2750, alpha=1e-7, solver='adam',
                                 verbose=10, random_state=42, tol=1e-9)
        self.clf.fit(X_correct, y_correct)

    def draw_digit(self, X_digit):
        """
        Draw a digit based on the given digit features.

        Args:
            X_digit (numpy.ndarray): Digit features.

        Returns:
            None
        """
        digit = self.clf.predict([X_digit])[0]
        plt.imshow(X_digit.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f"Digit: {digit}")
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def run(self):
        """
        Run the digit classifier pipeline.

        Returns:
            None
        """
        X_train, X_test, y_train, y_test = self.load_digits_dataset()
        self.train_classifier(X_train, X_test, y_train, y_test)
        self.predict(X_test, y_test)
        self.plot_loss_and_accuracy()
        self.plot_sample_images(X_test, self.y_pred, y_test)
        self.save_predictions_to_csv("predictions.csv", y_test)
        predictions = self.load_predictions_from_csv("predictions.csv")
        self.collect_incorrect_predictions(predictions)
        if self.incorrect_predictions:
            print("Incorrect Predictions:\n", "==================")
            for prediction in self.incorrect_predictions:
                print(f"Predicted: {prediction[0]} Actual: {prediction[1]}")
            self.count_error_occurrences()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(self.error_counts.keys(), self.error_counts.values())
            ax.set_xlabel('Number')
            ax.set_ylabel('Frequency of Errors')
            ax.set_title('Frequency of Errors in Predictions')
            plt.show()
        else:
            print("All Predictions are Correct")

        # Learn to draw based on correctly identified digits
        correct_predictions = [(pred, actual) for pred, actual in predictions if pred == actual]
        X_correct = [X_test[i] for i in range(len(X_test)) if (self.y_pred[i], y_test[i]) in correct_predictions]
        y_correct = [self.y_pred[i] for i in range(len(X_test)) if (self.y_pred[i], y_test[i]) in correct_predictions]
        self.learn_to_draw(X_correct, y_correct)
        print("Learning to draw based on correctly identified digits...")

        # Draw a number entered by the user
        while True:
            user_input = input("Enter a number (0-9) to draw (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            elif not user_input.isdigit() or int(user_input) < 0 or int(user_input) > 9:
                print("Invalid input. Please enter a number between 0 and 9.")
                continue
            else:
                digit = int(user_input)
                X_digit = X_correct[y_correct.index(digit)]
                self.draw_digit(X_digit)

if __name__ == '__main__':
    classifier = DigitClassifier()
    classifier.run()
