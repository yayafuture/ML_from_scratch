import numpy as np

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

class KNN():
    def __init__(self, k=5, dist=euclidean_distance):
        self.k = k
        self.dist = dist

    def predict(self, X_test, X_train, y_train):
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        y_predict = np.zeros(test_size)

        for i in range(test_size):
            distances = np.array([[self.dist(X_test[i], X_train[j]), y_train[j]] for j in range(train_size)])
            k_nearest_neighbors = distances[distances[:, 0].argsort()][:self.k]
            counts = np.bincount(k_nearest_neighbors[:, 1].astype('int'))
            y_predict[i] = counts.argmax()

        return y_predict
