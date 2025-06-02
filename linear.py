import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load in training data
def load_data():

    # Ask user for file name
    file_path = input("Enter the .csv data file name: ")
    file_path = f"Data/{file_path}.csv"

    return pd.read_csv(file_path)

# Make X and Y data matrix
def matrix_maker(featureA, featureB, m, data):

    # First Create m by 1 array (featureA = training feature)
    data_array = data[[featureA]].to_numpy()
    array_of_1s = np.ones([1, m]) # Create array of 1s

    # Add 1s to make X matrix
    X = np.insert(data_array, 0, array_of_1s, axis=1)

    # Create Y matrix
    Y = data[[featureB]].to_numpy()

    return X, Y

# Calculate weights and cost for the cost function
def calculate_cost(X, Y, m):
    A = np.linalg.pinv(np.dot(X.T, X))
    B = np.dot(X.T, Y)
    W = np.dot(A, B)

    # Caclulate J(w0, w1)
    A = np.dot(X, W) - Y
    J = (1/m) * np.dot(A.T, A)

    return J, W

# Plot the regression line
def plot_regression(X, Y, featureA, featureB, J, W):
    print("J: ", J)
    print("W0: ", W[0])
    print("W1: ", W[1])
    regression_line = W[0] + W[1] * X

    plt.figure()
    plt.scatter(X, Y, color="blue", label="Training Data")
    plt.plot(X, regression_line, color="red", label="Regression Line")
    plt.title(f"{featureA} vs. {featureB}")
    plt.xlabel(featureA)
    plt.ylabel(featureB)
    plt.legend(loc="best")
    plt.show()

# MAIN    
data = load_data()
m1 = len(data)
featureA1 = data.columns[0]
featureB1 = data.columns[1]
x1, y1 = matrix_maker(featureA1, featureB1, m1, data)
j1, w1 = calculate_cost(x1, y1, m1)
plot_regression(x1[:, 1], y1, featureA1, featureB1, j1, w1)
