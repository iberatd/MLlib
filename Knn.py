import numpy as np
from scipy.spatial.distance import cdist


class KnnClassifier:
    def __init__(self):
        self.data=0
        self.labels=[]


    def fit(self, X, y):

        self.data=X
        self.labels=np.array(y)

    def predict(self, X):
        dist = cdist(self.data, X)

        return dist