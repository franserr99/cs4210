# -------------------------------------------------------------------------
# AUTHOR: Francisco Serrano
# FILENAME: perception.py
# SPECIFICATION: Read the file optdigits.tra to build a
#    Single Layer Perceptron and a Multi-Layer Perceptron classifiers, find best set of parameters for both of them.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1hr
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
# pip install scikit-learn==0.18.rc2 if needed
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import math

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# reading the data by using Pandas library
df = pd.read_csv('optdigits.tra', sep=',', header=None)

# getting the first 64 fields to form the feature data for training
X_training = np.array(df.values)[:, :64]
# getting the last field to form the class label for training
y_training = np.array(df.values)[:, -1]

# reading the data by using Pandas library
df = pd.read_csv('optdigits.tes', sep=',', header=None)

# getting the first 64 fields to form the feature data for test
X_test = np.array(df.values)[:, :64]
# getting the last field to form the class label for test
y_test = np.array(df.values)[:, -1]

classifiers = ["Perceptron", "MLPClassifier"]

geoemtric_mean = int(math.sqrt(64 * 10))
max_accuracy_perceptron = 0
max_accuracy_mlp = 0

for learning_rate in n:  # iterates over n
    for shuffling in r:  # iterates over r
        for classifier in classifiers:  # iterates over the algorithms

            # Create a Neural Network classifier
            if classifier == "Perceptron":
                clf = Perceptron(shuffle=shuffling,
                                 eta0=learning_rate, max_iter=1000)
            else:
                clf = MLPClassifier(
                    activation='logistic',
                    learning_rate_init=learning_rate,
                    shuffle=shuffling,
                    max_iter=1000, hidden_layer_sizes=geoemtric_mean)

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)
            hits = 0
            rest = 0
            # make the classifier prediction for each test sample and start computing its accuracy
            # hint: to iterate over two collections simultaneously with zip() Example:
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])

                if (prediction == y_testSample):
                    hits += 1
                else:
                    rest += 1
            accuracy = hits / (hits+rest)
            if classifier == "Perceptron":
                if (accuracy > max_accuracy_perceptron):
                    max_accuracy_perceptron = accuracy
                    print(
                        f"Highest Perceptron accuracy so far: {accuracy}, Parameters: learning rate={learning_rate}, shuffle={shuffling}")

            else:
                if (accuracy > max_accuracy_mlp):
                    max_accuracy_mlp = accuracy
                    print(
                        f"Highest MLP accuracy so far: {accuracy}, Parameters: learning rate={learning_rate}, shuffle={shuffling}")
