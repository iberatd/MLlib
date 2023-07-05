import numpy as np
from Gradient import GradientDescent


class LogisticRegression:
    def __init__(self):
        self.w =0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        J = -1/m * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
        grad = 1/m * np.dot(X.T, (h - y))
        return J, grad
    
    def fit(self, X, y):
        gd = GradientDescent(cost_function=self.cost_function, num_iterations=1000)

        if(len(X.shape) == 1):
            X=X[:, np.newaxis]

        
        X = np.insert(X, 0, 1, axis=1)
        self.w , cost = gd.fit(X, y)

        return cost

    def predict(self, X):
        if(len(X.shape) == 1):
            X=X[:, np.newaxis]

        X = np.insert(X, 0, 1, axis=1)

        return self.sigmoid(np.dot(X[:, np.newaxis], self.w))
        