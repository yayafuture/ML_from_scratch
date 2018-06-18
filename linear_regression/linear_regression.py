import numpy as np

class l1():
    # alpha * sum of abs
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        loss = np.sum(np.fabs(w))
        return self.alpha * loss

    def grad(self, w):
        return self.alpha * np.sign(w)

class l2():
    # 0.5 * alpha * sum of squares
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        loss = w.T.dot(w)
        return self.alpha * 0.5 * float(loss)

    def grad(self, w):
        return self.alpha * w

class l1_l2():
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.sum(np.fabs(w))
        l2_contr = (1 - self.l1_ratio) * 0.5 * float(w.T.dot(w))
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)

class LinearRegression():
    """
    Parameters:
    -----------
    n_iterations: int
    learning_rate: float
    regularization: l1_regularization or l2_regularization or None
    gradient: Bool
        if use gradient descent to get result (if use regularization, can only use gradient descent)
    """

    def __init__(self, n_iterations=3000, learning_rate=0.00005, regularization=None, gradient=True):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        if regularization is None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def initialize_weights(self, n_features):
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = []
        if self.gradient:
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                loss = np.mean(0.5 * (y_pred - y) ** 2) + self.regularization(self.w)
                self.training_errors.append(loss)
                w_grad = X.T.dot(y_pred - y) + self.regularization.grad(self.w)  # (y_pred - y).T.dot(X)
                self.w = self.w - self.learning_rate * w_grad
        else:
            # normal equation
            X = np.matrix(X)
            y = np.matrix(y)
            X_T_X = X.T.dot(X)
            X_T_X_I_X_T = X_T_X.I.dot(X.T)
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.w = X_T_X_I_X_T_X_T_y

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
