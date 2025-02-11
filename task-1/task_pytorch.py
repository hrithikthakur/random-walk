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
parser.add_argument("--dist", choices=["cosine", "l2", "dot", "manhattan"], default="cosine", help="Select what distance measure to use: cosine, l2, dot, manhattan")
args = parser.parse_args()
device = args.device

print(f"Using device: {device}")

# ------------------------------------------------------------------------------------------------
# Task 1.1: Distance Functions (Implemented with GPU acceleration)
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    return 1 - torch.dot(X.to(device), Y.to(device)) / (torch.norm(X.to(device)) * torch.norm(Y.to(device)))

def distance_l2(X, Y):
    return torch.norm(X.to(device) - Y.to(device), p=2)

def distance_dot(X, Y):
    return torch.dot(X.to(device), Y.to(device))

def distance_manhattan(X, Y):
    return torch.norm(X.to(device) - Y.to(device), p=1)

# ------------------------------------------------------------------------------------------------
# Task 1.2: KNN Top-K Algorithm (Efficient GPU Implementation)
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K, distance_fn):
    A_torch = torch.tensor(A, device=device)
    X_torch = torch.tensor(X, device=device)
    
    distances = []
    for i in range(N):
        dist = distance_fn(A_torch[i], X_torch)
        distances.append((dist, i))
    
    distances.sort(key=lambda x: x[0])  # Sort by distance
    top_k_indices = [idx for _, idx in distances[:K]]
    
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Task 2.1: KMeans Algorithm (GPU-Accelerated)
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, distance_fn, max_iter=100, tol=1e-4):
    A_torch = torch.tensor(A, device=device)
    centroids = A_torch[torch.randperm(N)[:K]]
    
    for _ in range(max_iter):
        distances = torch.stack([torch.tensor([distance_fn(A_torch[i], c) for i in range(N)], device=device) for c in centroids])
        labels = torch.argmin(distances, dim=0)
        
        new_centroids = torch.stack([A_torch[labels == k].mean(dim=0) for k in range(K)])
        
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
    test_kmeans()
    test_knn()
    test_ann()