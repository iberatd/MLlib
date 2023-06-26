import numpy as np
import matplotlib.pyplot as plt 

class kMeansCluster:
    def __init__(self, max_iter=100):
        self.centroids = 0
        self.k=0
        self.max_iter = max_iter


    def fit(self, data, k=3):
        
        self.k=k

        f = data.shape[1] ##feature number

        self.centroids = np.random.uniform(0, 1, (k, f))

        mins = data.min(axis=0)
        maxs = data.max(axis=0)

        self.centroids = self.centroids* (maxs-mins) + mins

        fig, ax = plt.subplots(1, 5, figsize=(25,5))

        for i in range(self.max_iter):
            preds = self.predict(data)

            # ax[i].scatter(x=data[:,0], y= data[:,1], c= preds)
            # ax[i].scatter(self.centroids[:,0], self.centroids[:,1], c="r")

            old_cent = self.centroids.copy()
            

            self.update_centroids(data, preds)

            if(np.all(old_cent==self.centroids)):
                return self.centroids
            

            # ax[i].scatter(self.centroids[:,0], self.centroids[:,1], c="b")


        return self.centroids
    

    def predict(self, data):

        distances = np.ones((data.shape[0], self.k))

        for c in range(self.k):
            distances[:,c]*=((data-self.centroids[c])**2).sum(axis=1)

        labels = distances.argmin(axis=1)

        return labels
    
    def update_centroids(self, data, labels):

        num_classes = self.k

        # Create the identity matrix
        identity_matrix = np.eye(num_classes)

        # One-hot encode the labels
        one_hot_encoded = identity_matrix[labels]


        for c in range(self.k):
            self.centroids[c] = np.nan_to_num(data.T@one_hot_encoded[:,c], nan=0) / one_hot_encoded[:,c].sum()
