import torch
import cupy as cp
import triton
import numpy as np
import time
import json
import argparse
from test import testdata_kmeans, testdata_knn, testdata_ann

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Select device: cpu or cuda")
parser.add_argument("--dist", choices=["cosine", "l2", "dot", "manhattan"], default="cosine", help="Select what distance measure to use: cosine, l2, dot, manhattan")
args = parser.parse_args()
device = args.device

print(f"Using device: {device}")
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    pass

def distance_l2(X, Y):
    pass

def distance_dot(X, Y):
    pass

def distance_manhattan(X, Y):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------
def process_distance_func(arg):
    if arg == "cosine":
        return distance_cosine
    elif arg == "l2":
        return "distance_l2"
    elif arg == "dot":
        return distance_dot
    elif arg == "manhattan":
        return distance_manhattan
    else:
        return "ERROR: unknow distance function specified"

def test_kmeans():
    #N, D, A, K = testdata_kmeans("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, K = testdata_kmeans("")
    print("K-Means (task 1) results are:")
    kmeans_result = our_kmeans(N, D, A, K, process_distance_func(args.dist))
    print(kmeans_result)

def test_knn():
    #N, D, A, X, K = testdata_knn("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K, process_distance_func(args.dist))
    print("KNN (task 1) results are:")
    print(knn_result)

def test_ann():
    #N, D, A, X, K = testdata_ann("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, X, K = testdata_ann("")
    ann_result = our_ann(N, D, A, X, K, process_distance_func(args.dist))
    print("ANN (task 2.2) results are:")
    print(ann_result)

def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    # Define tests to run.
    tests = [test_kmeans, test_knn, test_ann]

    # Run tests and measure time taken.
    for callable in tests:
        start = time.time()
        callable()
        print(f"=== {callable.__name__}() completed in {time.time() - start:.4f} seconds ===\n")
