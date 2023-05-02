# Neural-Network-Digit-Classifier

The "Neural-Network-Digit-Classifier" is an implementation of a multilayer perceptron (MLP) neural network to classify handwritten digits from the popular MNIST dataset. The program imports necessary libraries, loads the dataset, and splits it into training and testing sets. It then trains the MLP classifier on the training set and calculates its accuracy on the testing set.

The MLP neural network is a popular type of artificial neural network that consists of multiple layers of interconnected neurons. It is a type of feedforward neural network where information flows only in one direction, from the input layer to the output layer. The MLP used in this program has two hidden layers of 50 neurons each and uses the Adam optimizer to minimize the cross-entropy loss function.

The program records the accuracy and loss of the MLP classifier on both the training and testing sets over time using the partial_fit() method, which updates the weights of the neural network incrementally as new data points are fed to it. It then plots the accuracy and loss curves using the Matplotlib library. The program also displays sample images from the dataset with their predicted labels to give the user a visual representation of the MLP classifier's performance.

Example Predictions:

![Figure_1 8](https://user-images.githubusercontent.com/94536625/234828717-a947937c-6384-4a76-b3d5-294d69669757.png)

