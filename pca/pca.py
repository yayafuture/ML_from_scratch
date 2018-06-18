import numpy as np
from sklearn.preprocessing import StandardScaler

class PCA():
    def _calculate_covariance_matrix(self, X, Y=None):
        m = X.shape[0]
        X = StandardScaler().fit_transform(X)
        Y = X if Y is None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    def transform(self, X, n_components, Y=None):
        """ change the dimension to n_components"""
        covariance_matrix = self._calculate_covariance_matrix(X, Y)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # get the eigenvectors corresponding to the largest n_components eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]
        # return dimension: m * n_components
        return np.matmul(X, eigenvectors)
