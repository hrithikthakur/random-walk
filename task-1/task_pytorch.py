import torch
#import cupy as cp
#import triton
#import triton.language as tl
import numpy as np
import time
import json
import argparse
from test import testdata_kmeans, testdata_knn, testdata_ann
#Note: probably don't need every import in every version of the code

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Select device: cpu or cuda")
parser.add_argument("--dist", choices=["cosine", "l2", "dot", "manhattan"], default="l2", help="Select what distance measure to use: cosine, l2, dot, manhattan")
args = parser.parse_args()
device = args.device

print(f"Using device: {device}")

# ------------------------------------------------------------------------------------------------
# Task 1.1: Distance Functions (Implemented with GPU acceleration)
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    return 1 - (torch.sum(X * Y) / (torch.sqrt(torch.sum(X ** 2)) * torch.sqrt(torch.sum(Y ** 2))))

def distance_l2(X, Y):
    return torch.norm(X - Y, p=2)

def distance_dot(X, Y):
    return torch.dot(X, Y)

def distance_manhattan(X, Y):
    return torch.norm(X - Y, p=1)

# ------------------------------------------------------------------------------------------------
# Task 1.2: KNN Top-K Algorithm (Efficient GPU Implementation)
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K, distance_fn):
    A_torch = torch.tensor(A, device=device).reshape(N, D)
    X_torch = torch.tensor(X, device=device)

    distances = torch.vmap(distance_fn, in_dims=(0, None))(A_torch, X_torch)

    _, top_k_indices = torch.topk(distances, K, largest=False)

    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Task 2.1: KMeans Algorithm (GPU-Accelerated)
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, distance_fn, max_iter=100, tol=1e-4):
    A_torch = torch.tensor(A, device=device).reshape(N, D)
    centroids = A_torch[torch.randperm(N)[:K]] #Chooses K initial points as random centroids
    labels = torch.zeros(N, device=device, dtype=torch.long) #Initialise a list full of zeros for the labels

    # Vectorized distance function (This is the secret sauce, takes the vector based distance function and makes it apply in batches with matrices)
    # vmap is basically a replacement for python loops, but in uses atching computations, which allows PyTorch to process multiple inputs in parallel
    distance_vmap = torch.vmap(distance_fn, in_dims=(None, 0))  # Batch over centroids

    for _ in range(max_iter):
        # Assignment step
        # NOTE: you can look at the double vmap as a nested loop "for each centroid: for each point: calculate distance"
        distances = torch.vmap(distance_vmap, in_dims=(0, None))(A_torch, centroids)  #TODO: can it be done with just 1 vmap?    
        labels = torch.argmin(distances, dim=1)  # Assign to closest centroid
        
        #Update centroids
        new_centroids = torch.stack([A_torch[labels == k].mean(dim=0) if (labels == k).any() else centroids[k] for k in range(K)]) #TODO: can python loop be avoided here?
        
        #Check convergence
        if torch.norm(new_centroids - centroids) < tol:
            break
    
        centroids = new_centroids
    return labels.cpu().numpy()


def our_ann(N, D, A, X, K, distance_fn, K_clusters=10):
    labels = our_kmeans(N, D, A, K_clusters, distance_fn=distance_fn)

    X_torch = torch.tensor(X, device=device)
    A_torch = torch.tensor(A, device=device)
    labels_torch = torch.tensor(labels, device=device)

    distances = []
    for i in range(N):
        dist = distance_fn(A_torch[i], X_torch)
        distances.append((dist, i, labels[i]))

    distances.sort(key=lambda x: x[0])

    cluster_id = distances[0][2]
    cluster_indices = [idx for _, idx, lbl in distances if lbl == cluster_id]

    top_k_indices = cluster_indices[:K]
    return top_k_indices


# ------------------------------------------------------------------------------------------------
# Test the Implementations
# ------------------------------------------------------------------------------------------------
def process_distance_func(arg):
    if arg == "cosine":
        return distance_cosine
    elif arg == "l2":
        return distance_l2
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
