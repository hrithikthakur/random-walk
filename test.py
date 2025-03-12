import numpy as np
import json

def read_data(file_path=""):
    if file_path == "":
        return None
    if file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def ensure_2d(X, D):
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X

def testdata_kmeans(test_file):
    if test_file == "":
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        K = 10
        return N, D, A, K
    else:
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            K = data["k"]
            A = read_data(A_file)
        return N, D, A, K

def testdata_knn(test_file, M=1):
    if test_file == "":
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(M, D)  # Already 2D with M query points
        K = 10
        return N, D, A, X, K
    else:
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            A = read_data(A_file)
            X = read_data(X_file)
            X = ensure_2d(X, D)  # Ensure X is 2D
        return N, D, A, X, K
    
def testdata_ann(test_file, M=1):
    if test_file == "":
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(M, D)  # Already 2D with M query points
        K = 10
        return N, D, A, X, K
    else:
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            A = read_data(A_file)
            X = read_data(X_file)
            X = ensure_2d(X, D)  # Ensure X is 2D
        return N, D, A, X, K 