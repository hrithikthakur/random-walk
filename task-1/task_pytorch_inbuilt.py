# NOTE: this implementation is faster, but is using inbuilt pytorch functions, so probably not very good :/
# NOTE: useful as a target though

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


def our_knn(N, D, A, X, K):
    A_torch = torch.tensor(A, device=device)
    X_torch = torch.tensor(X, device=device)

    distances = torch.cdist(A_torch, X_torch.unsqueeze(0), p=2).squeeze()
    top_k_indices = torch.topk(distances, K, largest=False).indices.cpu().numpy()
    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Task 2.1: KMeans Algorithm (GPU-Accelerated)
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K):
    max_iter = 100
    tol = 1e-4
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



def our_ann(N, D, A, X, K, device='cpu'):
    # Convert inputs to PyTorch tensors
    A_torch = torch.as_tensor(A, device=device).clone().detach().reshape(N, D)
    X_torch = torch.as_tensor(X, device=device).reshape(1, D)  # Ensure X is 2D for cdist

    # Assuming our_kmeans is a function that returns cluster labels
    labels = torch.tensor(our_kmeans(N, D, A_torch, K), device=device)  # Convert labels to PyTorch tensor

    # Compute centroids
    centroids = torch.stack([A_torch[labels == k].mean(dim=0) if (labels == k).any() else torch.zeros(D, device=device) for k in range(K)])
    
    # Step 2: Find the nearest cluster center to X
    centroid_distances = torch.cdist(X_torch, centroids).squeeze(0)  # Pairwise distances between X and centroids
    nearest_cluster_index = torch.argmin(centroid_distances)

    # Step 3: Find the second nearest cluster center (different from the nearest)
    centroid_distances[nearest_cluster_index] = float('inf')  # Mask the nearest cluster
    second_nearest_cluster_index = torch.argmin(centroid_distances)

    # Step 4: Merge vectors from both clusters (K1 and K2)
    cluster_1_indices = (labels == nearest_cluster_index).nonzero(as_tuple=True)[0]
    cluster_2_indices = (labels == second_nearest_cluster_index).nonzero(as_tuple=True)[0]
    candidate_indices = torch.cat([cluster_1_indices, cluster_2_indices])

    # Step 5: Find the top-K nearest neighbors among these candidates
    candidate_vectors = A_torch[candidate_indices]
    candidate_distances = torch.cdist(X_torch, candidate_vectors).squeeze(0)  # Pairwise distances between X and candidates
    _, top_k_indices = torch.topk(candidate_distances, K, largest=False)

    # Return final top-K nearest neighbor indices
    return candidate_indices[top_k_indices]
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
    print("K-Means (task 2.1) results are:")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    #N, D, A, X, K = testdata_knn("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K)
    print("KNN (task 1) results are:")
    print(knn_result)

def test_ann():
    #N, D, A, X, K = testdata_ann("test_file.json") #TODO: aquire or create a JSON file
    N, D, A, X, K = testdata_ann("")
    ann_result = our_ann(N, D, A, X, K)
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
