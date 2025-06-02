"""
@Author: Sai Yadavalli
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to load in training data
def load_data():

    # Ask user for file name
    file_path = input("Enter the .csv data file name: ")
    file_path = f"Data/{file_path}.csv"
    
    # Read data and print column names
    df = pd.read_csv(file_path)
    print("The data headers are: ", list(df))

    return df

# Function to make X and Y data matrix
def matrix_maker(trainFeatures, featureB, m, data):

    # First Create m by n array (n = number of features)
    data_array = data[trainFeatures].to_numpy()
    array_of_1s = np.ones([1, m]) # Create array of 1s

    # Add 1s to make X matrix
    X = np.insert(data_array, 0, array_of_1s, axis=1)

    # Create Y matrix
    Y = data[featureB].to_numpy()

    return X, Y

# Function to calculate weights and cost for the cost function
def calculate_cost(X, Y, m):
    A = np.linalg.pinv(np.dot(X.T, X))
    B = np.dot(X.T, Y)
    W = np.dot(A, B)

    # Caclulate J(w0, w1)
    A = np.dot(X, W) - Y
    J = (1/m) * np.dot(A.T, A)

    print("J: ", J)
    for i in range(len(W)):
        print(f"W{i}: ", W[i])

    return J, W

# Function to plot the regression line and data
def plot_regression(X, Y, featureA, featureB, W):

    # Calculate the regression line
    regression_line = np.dot(X, W)

    # Plot the figures and regression line
    plt.figure()
    plt.scatter(X[:, 1], Y, color="blue", label="Training Data")
    plt.plot(X[:, 1], regression_line, color="red", label="Regression Line")
    plt.title(f"{featureA} vs. {featureB}")
    plt.xlabel(featureA)
    plt.ylabel(featureB)
    plt.legend(loc="best")
    plt.show()
        
# MAIN
data = load_data()
m1 = len(data)
y_feature = input("Enter the dependent variable: ")
featureB1 = y_feature
trainFeatures1 = data.columns.drop(y_feature)
x1, y1 = matrix_maker(trainFeatures1, featureB1, m1, data)
j1, w1 = calculate_cost(x1, y1, m1)
plot_regression(x1, y1, trainFeatures1, featureB1, w1)
