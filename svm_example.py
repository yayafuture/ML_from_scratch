import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from pca.pca import *
from support_vector_machine.kernels import *
from support_vector_machine.svmModel import *
logging.basicConfig(level=logging.DEBUG)

def run():
    start = time.clock()
    X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
                               random_state=1111, n_classes=2, class_sep=1.75, )
    # {0,1} to {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # set kernel
    kernel = RBF(gamma=0.1)
    model = SVM(X_train, y_train, max_iter=500, kernel=kernel, C=0.6)
    model.train()

    predictions = model.predict(X_test)
    accuracyRate = accuracy(y_test, predictions)
    print('Classification accuracy: %s' % accuracyRate)

    # original data
    #plot_in_2d(X_test, y_test, title="Support Vector Machine", accuracy=accuracyRate)

    # classification effect
    plot_in_2d(X_test, predictions, title="Support Vector Machine", accuracy=accuracyRate)

    end = time.clock()
    print("Time: %fs" % (end - start))

if __name__ == '__main__':
    run()
