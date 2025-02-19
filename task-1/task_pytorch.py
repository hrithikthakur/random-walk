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
    distance_fn = distance_fn = process_distance_func(args.dist)
    A_torch = torch.tensor(A, device=device).reshape(N, D)
    X_torch = torch.tensor(X, device=device)

    distances = torch.vmap(distance_fn, in_dims=(0, None))(A_torch, X_torch)

    _, top_k_indices = torch.topk(distances, K, largest=False)

    return top_k_indices

# ------------------------------------------------------------------------------------------------
# Task 2.1: KMeans Algorithm (GPU-Accelerated)
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K):
    distance_fn = process_distance_func(args.dist)
    max_iter = 100
    tol = 1e-4
    A_torch = torch.as_tensor(A, device=device).clone().detach().reshape(N, D)
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
    return labels


def our_ann(N, D, A, X, K):
    distance_fn = process_distance_func(args.dist)

    A_torch = torch.as_tensor(A, device=device).clone().detach().reshape(N, D)
    X_torch = torch.tensor(X, device=device)

    #Run K-Means to get labels
    labels = our_kmeans(N, D, A_torch, K)

    # Compute centroids
    #TODO: check if we can avoid the python loop here
    #TODO: check if we can return centroids from the k-means to avoid recalculation (ask TAs if we can edit returns)
    centroids = torch.stack([A_torch[labels == k].mean(dim=0) if (labels == k).any() else torch.zeros(D, device=device) for k in range(K)])
    
    # Step 2: Find the nearest K1 cluster centers to X
    centroid_distances = torch.vmap(distance_fn, in_dims=(0, None))(centroids, X_torch)
    #_, nearest_cluster_indices = torch.topk(centroid_distances, K1, largest=False)
    nearest_cluster_index = torch.argmin(centroid_distances)
    
    # Step 3: Find the **second nearest (K2=1) cluster center**, different from K1
    #TODO: not sure where KNN comes in here.... Not sure if this is correct...
    centroid_distances = torch.vmap(distance_fn, in_dims=(0, None))(centroids, centroids[nearest_cluster_index])
    _, top2_indices = torch.topk(centroid_distances, k=2, largest=False)
    second_nearest_cluster_index = top2_indices[1]

    # Step 4: Merge vectors from **both clusters (K1 and K2)**
    cluster_1_indices = torch.nonzero(labels == nearest_cluster_index, as_tuple=True)[0]
    cluster_2_indices = torch.nonzero(labels == second_nearest_cluster_index, as_tuple=True)[0]
    candidate_indices = torch.cat([cluster_1_indices, cluster_2_indices])

    # Find the top-K nearest neighbors among these candidates
    candidate_vectors = A_torch[candidate_indices]
    candidate_distances = torch.vmap(distance_fn, in_dims=(0, None))(candidate_vectors, X_torch)
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
