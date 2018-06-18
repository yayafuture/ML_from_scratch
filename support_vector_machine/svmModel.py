import numpy as np
import matplotlib.pyplot as plt
import progressbar
from utils.misc import bar_widgets
import logging
from pca.pca import PCA
from support_vector_machine.kernels import *

class SVM(object):
    def __init__(self, trainX, trainY, C=1, kernel=None, difference=1e-3, max_iter=100):
        self.X = trainX
        self.Y = trainY
        self.m = trainX.shape[0]
        self.n = trainX.shape[1]
        self.difference = difference # if converge
        self.max_iter = max_iter

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        self.C = C # regularization parameter
        self.b = 0 # bias
        self.alpha = np.zeros(self.m) # lagrange multiplier

        # features after using kernel function, K[i, j] is the result of kernel(Xi, Xj)
        self.K = np.zeros((self.m, self.m))
        if not kernel:
            self.kernel = LinearKernel()
        else:
            self.kernel = kernel
        for i in range(self.m):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])

    def train(self):
        for now_iter in self.bar(range(self.max_iter)):
            alpha_prev = np.copy(self.alpha)
            for j in range(self.m): # alpha2, select the alpha that violate KKT
                # choose another lagrange multiplier different from j
                i = self._random_index(j)
                error_i, error_j = self._error_row(i), self._error_row(j)

                # Y*error is equal to Y*predict - 1, so compare Y*predict with 1 is equal to compare Y*error with 0
                if (self.Y[j] * error_j < -0.001 and self.alpha[j] < self.C) or (self.Y[j] * error_j > 0.001 and self.alpha[j] > 0):
                    # Lihang equation 7.107
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]

                    if eta >= 0:
                        continue

                    L, H = self._getBounds(i, j)
                    old_alpha_j, old_alpha_i = self.alpha[j], self.alpha[i]
                    # Lihang equation 7.106, best alpha2 without restriction
                    self.alpha[j] -= (self.Y[j] * (error_i - error_j)) / eta
                    # Lihang equation 7.108, alpha2 after restriction
                    self.alpha[j] = self._finalValue(self.alpha[j], H, L)
                    self.alpha[i] = self.alpha[i] + self.Y[i] * self.Y[j] * (old_alpha_j - self.alpha[j])

                    # update b
                    b1 = self.b - error_i - self.Y[i] * (self.alpha[i] - old_alpha_j) * self.K[i, i] - \
                         self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[i, j]
                    b2 = self.b - error_j - self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[j, j] - \
                         self.Y[i] * (self.alpha[i] - old_alpha_i) * self.K[i, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

            # if converge
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.difference:
                break

    # random choose one lagrange multiplier, must be different from first_alpha
    def _random_index(self, first_alpha):
        i = first_alpha
        while i == first_alpha:
          i = np.random.randint(0, self.m - 1)
        return i

    # calculate wx+b, w is represented as in Lihang's book equation 7.56
    def predict_row(self, X):
        k_v = self.kernel(self.X, X)
        return np.dot((self.alpha * self.Y).T, k_v.T) + self.b

    # return value: 1 for positive and -1 for negative
    def predict(self, X):
        n = X.shape[0]
        result = [np.sign(self.predict_row(X[i, :])) for i in range(n)]
        return result

    def _error_row(self, i):
        return self.predict_row(self.X[i]) - self.Y[i]

    # get the range for self.alpha[j], Lihang page 126
    def _getBounds(self, i, j):
        if self.Y[i] != self.Y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H

    # best alpha2 in restriction
    def _finalValue(self, alpha, H, L):
        if alpha > H:
            return H
        if alpha < L:
            return L
        return alpha

def accuracy(actual, predicted):
    return 1.0 - sum(actual != predicted) / float(actual.shape[0])

# use pca module
def plot_in_2d(X, y=None, title=None, accuracy=None, legend_labels=None):

    cmap = plt.get_cmap('viridis')
    X_transformed = PCA().transform(X, 2)

    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    class_distr = []
    y = np.array(y).astype(int)

    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Plot legend
    if not legend_labels is None:
        plt.legend(class_distr, legend_labels, loc=1)

    # Plot title
    if title:
        if accuracy:
            perc = 100 * accuracy
            plt.suptitle(title)
            plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
        else:
            plt.title(title)

    # Axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
