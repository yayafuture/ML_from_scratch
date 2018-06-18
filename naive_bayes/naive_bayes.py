import numpy as np

class NaiveBayes():
    # assume p(Xi|y) is normal distribution
    def fit(self, X_trian, y_trian):
        self.X = X_trian
        self.m = self.X.shape[0]
        self.y = y_trian
        self.classes = np.unique(self.y)
        self.parameters = {}
        for i, c in enumerate(self.classes):
            # calculate the mean, variance of X in each class, and the prior
            X_c = self.X[np.where(self.y == c)]
            X_c_mean = np.mean(X_c, axis=0, keepdims=True)
            X_c_var = np.var(X_c, axis=0, keepdims=True)
            parameter = {"mean": X_c_mean, "var": X_c_var, "prior": X_c.shape[0] / self.m}
            self.parameters["class " + str(c)] = parameter

    def _pdf(self, X, _class):
        # assume Gaussian distribution
        # eps is used to avoid divide 0
        eps = 1e-4
        mean = self.parameters["class " + str(_class)]["mean"]
        var = self.parameters["class " + str(_class)]["var"]

        numerator = np.exp(-(X - mean) ** 2 / (2 * var + eps))
        denominator = np.sqrt(2 * np.pi * var + eps)

        # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
        result = np.sum(np.log(numerator / denominator), axis=1, keepdims=True)
        return result.T

    def _predict(self, x):
        # for one instance, get the probability of each class
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters["class " + str(y)]["prior"])
            posterior = self._pdf(x, y)
            prediction = prior + posterior
            output.append(prediction)
        return np.array(output)

    def predict(self, x):
        # return the class with largest probability for x.shape[0] instances
        result = []
        for i in range(x.shape[0]):
            output = self._predict(x[i, :])
            result.append(np.argmax(output))
        return np.array(result)
