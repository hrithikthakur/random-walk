import torch
import cupy as cp
import triton
import numpy as np
import time
import json
import argparse
from test import testdata_kmeans, testdata_knn, testdata_ann

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Select device: cpu or cuda")
parser.add_argument("--dist", choices=["cosine", "l2", "dot", "manhattan"], default="cosine", help="Select what distance measure to use: cosine, l2, dot, manhattan")
args = parser.parse_args()
device = args.device

print(f"Using device: {device}")

def distance_cosine(X, Y):
    X_norm = cp.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True)
    similarity = cp.dot(X, Y.T) / (X_norm @ Y_norm.T + 1e-8)
    return 1 - similarity

def distance_l2(X, Y):
    XX = cp.sum(X ** 2, axis=1, keepdims=True)
    YY = cp.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * cp.dot(X, Y.T)
    distances = cp.maximum(distances, 0)
    return cp.sqrt(distances)

def distance_l2_inefficient(X, Y):
    X_expanded = cp.expand_dims(X, axis=1)
    Y_expanded = cp.expand_dims(Y, axis=0)
    distances = cp.sum((X_expanded - Y_expanded) ** 2, axis=2)
    return cp.sqrt(distances)

def distance_dot(X, Y):
    similarity = cp.dot(X, Y.T)
    return -similarity

def distance_manhattan(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = cp.zeros((n_x, n_y))
    for i in range(X.shape[1]):
        diff = cp.abs(X[:, i:i+1] - Y[:, i].reshape(1, -1))
        distances += diff
    return distances

def our_knn(N, D, A, X, K, distance_func):
    if A.shape[0] != N or A.shape[1] != D:
        raise ValueError(f"Matrix A should have shape ({N}, {D}), but got {A.shape}")
    if X.shape[1] != D:
        raise ValueError(f"Matrix X should have {D} dimensions, but got {X.shape[1]}")
    if K > N:
        raise ValueError(f"Cannot find {K} nearest neighbors with only {N} database points")
    
    A = cp.asarray(A)
    X = cp.asarray(X)
    
    distances = distance_func(X, A)
    indices = cp.argsort(distances, axis=1)[:, :K]
    
    return cp.asnumpy(indices)

def our_kmeans(N, D, A, K, distance_func):
    if A.shape[0] != N or A.shape[1] != D:
        raise ValueError(f"Matrix A should have shape ({N}, {D}), but got {A.shape}")
    if K > N:
        raise ValueError(f"Cannot create {K} clusters with only {N} points")
    
    A = cp.asarray(A)
    centroids = A[cp.random.choice(N, K, replace=False)]
    max_iterations = 100
    prev_centroids = None
    
    for _ in range(max_iterations):
        distances = distance_func(A, centroids)
        labels = cp.argmin(distances, axis=1)
        new_centroids = cp.zeros_like(centroids)
        for k in range(K):
            mask = (labels == k)
            if cp.any(mask):
                new_centroids[k] = cp.mean(A[mask], axis=0)
            else:
                new_centroids[k] = centroids[k]
        
        if prev_centroids is not None and cp.allclose(new_centroids, prev_centroids):
            break
            
        prev_centroids = new_centroids.copy()
        centroids = new_centroids
    
    return cp.asnumpy(centroids)

def our_ann_lsh(N, D, A, X, K, distance_func):
    if A.shape[0] != N or A.shape[1] != D:
        raise ValueError(f"Matrix A should have shape ({N}, {D}), but got {A.shape}")
    if X.shape[1] != D:
        raise ValueError(f"Matrix X should have {D} dimensions, but got {X.shape[1]}")
    if K > N:
        raise ValueError(f"Cannot find {K} nearest neighbors with only {N} database points")
    
    A = cp.asarray(A)
    X = cp.asarray(X)
    
    if distance_func.__name__ != 'distance_cosine':
        return our_knn(N, D, A, X, K, distance_func)
    
    num_hashes = 50
    rp = cp.random.randn(D, num_hashes)
    db_hashes = cp.sign(A @ rp)
    query_hashes = cp.sign(X @ rp)
    hash_similarities = (query_hashes @ db_hashes.T) / num_hashes
    candidate_k = min(N, K * 10)
    candidate_indices = cp.argsort(-hash_similarities, axis=1)[:, :candidate_k]
    final_indices = cp.zeros((X.shape[0], K), dtype=cp.int64)
    
    for i in range(X.shape[0]):
        candidates = A[candidate_indices[i]]
        exact_distances = distance_cosine(X[i:i+1], candidates)
        final_indices[i] = candidate_indices[i][cp.argsort(exact_distances[0])[:K]]
    
    return cp.asnumpy(final_indices)

def our_ann_kmeans(N, D, A, X, K, distance_func):
    A = cp.asarray(A)
    X = cp.asarray(X)
    
    n_clusters = min(int(cp.sqrt(N).item()), N // 10)
    
    centroids = our_kmeans(N, D, A, n_clusters, distance_func)
    centroids = cp.asarray(centroids)
    
    db_distances = distance_func(A, centroids)
    db_labels = cp.argmin(db_distances, axis=1)
    
    final_indices = cp.zeros((X.shape[0], K), dtype=cp.int64)
    
    query_distances = distance_func(X, centroids)
    
    n_clusters_to_search = min(3, n_clusters)
    
    for i in range(X.shape[0]):
        nearest_clusters = cp.argsort(query_distances[i])[:n_clusters_to_search]
        
        candidate_indices = cp.array([], dtype=cp.int64)
        for cluster_idx in nearest_clusters:
            cluster_points = cp.where(db_labels == cluster_idx)[0]
            candidate_indices = cp.concatenate([candidate_indices, cluster_points])
        
        if len(candidate_indices) < K:
            remaining_points = cp.where(~cp.isin(cp.arange(N), candidate_indices))[0]
            candidate_indices = cp.concatenate([candidate_indices, remaining_points])
        
        candidates = A[candidate_indices]
        exact_distances = distance_func(X[i:i+1], candidates)
        
        nearest_k = cp.argsort(exact_distances[0])[:K]
        final_indices[i] = candidate_indices[nearest_k]
    
    return cp.asnumpy(final_indices)

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
        raise ValueError("Unknown distance function specified")
    
def test_kmeans():
    N, D, A, K = testdata_kmeans("")
    print("K-Means (task 1) results are:")
    kmeans_result = our_kmeans(N, D, A, K, process_distance_func(args.dist))
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K, process_distance_func(args.dist))
    print("KNN (task 1) results are:")
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("")
    ann_result = our_kmeans_ann(N, D, A, X, K, process_distance_func(args.dist))
    print("ANN (task 2.2) results are:")
    print(ann_result)
    
def recall_rate(list1, list2):
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
    test_knn()
    test_ann()