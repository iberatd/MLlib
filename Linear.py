import numpy as np

class LinearRegression:
    def __init__(self):
        self.w=[]

    
    def fit(self, X, y): 
        """format of X should be matrix"""


        if len(X.shape)==1:
            X= np.expand_dims(X, axis=1)

        if len(y.shape)==1:
            y= np.expand_dims(y, axis=1)

        
        # print(X)

        
        X = np.concatenate([  np.ones((X.shape[0],1)), X], axis=1)
        XT = X.T

        try:
            XTX_1 = np.linalg.inv( np.matmul(XT, X) )
        except:
            print("Cannot take inverse of the matrix")


        XTX_1XT = np.matmul(XTX_1, XT)

        self.w = np.matmul(XTX_1XT,y)

        # print(self.w)

    def predict(self, X):

        X = np.concatenate([  np.ones((X.shape[0],1)), X], axis=1)
        return np.matmul(X, self.w)


