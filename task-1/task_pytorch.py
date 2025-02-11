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

def our_knn(N, D, A, X, K):
    A_torch = torch.tensor(A, device=device)
    X_torch = torch.tensor(X, device=device)
    
    distances = torch.cdist(A_torch, X_torch.unsqueeze(0), p=2).squeeze()
    top_k_indices = torch.topk(distances, K, largest=False).indices.cpu().numpy()
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Task 2.1: KMeans Algorithm (GPU-Accelerated)
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, max_iter=100, tol=1e-4):
    A_torch = torch.tensor(A, device=device)
    centroids = A_torch[torch.randperm(N)[:K]]
    
    for _ in range(max_iter):
        distances = torch.cdist(A_torch, centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        
        new_centroids = torch.stack([A_torch[labels == k].mean(dim=0) for k in range(K)])
        
        if torch.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    
    return labels.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Task 2.2: Approximate Nearest Neighbor (ANN) with Clustering
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K, K_clusters=10):
    labels = our_kmeans(N, D, A, K_clusters)
    
    X_torch = torch.tensor(X, device=device)
    A_torch = torch.tensor(A, device=device)
    labels_torch = torch.tensor(labels, device=device)
    
    cluster_id = labels_torch[torch.argmin(torch.cdist(A_torch, X_torch.unsqueeze(0), p=2))]
    cluster_indices = (labels_torch == cluster_id).nonzero().squeeze()
    
    distances = torch.cdist(A_torch[cluster_indices], X_torch.unsqueeze(0), p=2).squeeze()
    top_k_indices = cluster_indices[torch.topk(distances, K, largest=False).indices].cpu().numpy()
    
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Test the Implementations
# ------------------------------------------------------------------------------------------------

def test_kmeans():
    #N, D, A, K = testdata_kmeans("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, K = testdata_kmeans("")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    #N, D, A, X, K = testdata_knn("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    #N, D, A, X, K = testdata_ann("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, X, K = testdata_knn("")
    ann_result = our_ann(N, D, A, X, K)
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