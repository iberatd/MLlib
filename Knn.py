import numpy as np
from scipy.spatial.distance import cdist


class KnnClassifier:
    def __init__(self, k):
        self.data=0
        self.labels=[]
        self.k=k


    def fit(self, X, y):

        self.data=X
        self.labels=np.array(y)

    def predict(self, X):
        dist = cdist(self.data, X)

        neighbours = np.argsort(dist, axis=0)[:self.k, :]

        n_labels = self.labels[neighbours]

        results = []

        for i in range(len(dist[0])):
            results.append([np.argmax(np.bincount(n_labels[:,i]))])

        return np.array(results)
    
class KnnRegressor:
    def __init__(self, k):
        self.data=0
        self.values=[]
        self.k=k


    def fit(self, X, y):

        self.data=X
        self.values=np.array(y)

    def predict(self, X):
        dist = cdist(self.data, X)

        neighbours = np.argsort(dist, axis=0)[:self.k, :]

        n_values = self.values[neighbours]

        return n_values.mean(axis=0)
    
    