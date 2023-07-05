import numpy as np

class GradientDescent:
    def __init__(self, cost_function, learning_rate = 0.01, num_iterations =100):
        """cost function should return cost and gradient"""
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    
    def fit(self, X, y):
        m = len(y)
        theta = np.zeros(X.shape[1])

        
        for i in range(self.num_iterations):
            cost , grad = self.cost_function(X, y, theta)
            theta = theta - self.learning_rate * grad

        return theta, cost