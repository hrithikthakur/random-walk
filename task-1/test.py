import numpy as np
import json

def read_data(file_path=""):
    """
    Read data from a file
    """
    if file_path == "":
        return None
    if file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def testdata_kmeans(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        K = 10
        return N, D, A, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            K = data["k"]
            # A = np.loadtxt(A_file) The original line
            A = read_data(A_file) #Edited line (TODO: discuss in meeting)
        return N, D, A, K

def testdata_knn(test_file):
    if test_file == "":
        print(f"TESTFILE IS {test_file}")
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(1, D)  # Changed this line to make X 2D
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            # A = np.loadtxt(A_file) The original line
            A = read_data(A_file) #Edited line (TODO: discuss in meeting) 
            #X = np.loadtxt(X_file) The original line
            X = read_data(X_file) #Edited line (TODO: discuss in meeting) 
        return N, D, A, X, K
    
def testdata_ann(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            # A = np.loadtxt(A_file) The original line
            A = read_data(A_file) #Edited line (TODO: discuss in meeting) 
            #X = np.loadtxt(X_file) The original line
            X = read_data(X_file) #Edited line (TODO: discuss in meeting) 
        return N, D, A, X, K
