from sklearn.datasets import fetch_openml
from sklearn.datasets import get_data_home
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import zero_one_loss
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#from sklearn.externals import joblib
import os
import numpy as np
import joblib 

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = joblib.Memory(os.path.join(get_data_home(),
                "mnist_benchmark_data"), mmap_mode="r")


@memory.cache
def load_data(dtype=np.float32, order="F"):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...\n")
    data = fetch_openml("mnist_784")
    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    estimator = make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=200))
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print(estimator.score(X_test, y_test))
