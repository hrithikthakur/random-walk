# import torch
import cupy as cp
# import triton
import numpy as np
import time
# import json
import argparse
from test import testdata_kmeans, testdata_knn, testdata_ann

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Select device: cpu or cuda")
parser.add_argument("--dist", choices=["cosine", "l2", "dot", "manhattan"], default="l2", help="Select what distance measure to use: cosine, l2, dot, manhattan")
parser.add_argument("--testfile", default="",
                    help="Specify test file JSON. Leave empty for a randomised small test")

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
    N, D, A, K = testdata_kmeans(args.testfile)
    kmeans_result = our_kmeans(N, D, A, K, process_distance_func(args.dist))
    print("K-Means (task 1.1) results are:")
    print(kmeans_result)
    
def test_kmeans_detailed():
    # test data
    N, D, A, K = testdata_kmeans(args.testfile)
    
    # initial setup
    print("\nK-Means Clustering Test:")
    print(f"Number of points (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Number of clusters (K): {K}")
    print(f"Data shape: {A.shape}")
    print(f"Distance metric: {args.dist}")
    
    # start time
    start_time = time.time()
    
    # run teh function
    kmeans_result = our_kmeans(N, D, A, K, process_distance_func(args.dist))
    
    # net execution time
    execution_time = time.time() - start_time
    
    print("\nResults:")
    print(f"Centroids shape: {kmeans_result.shape}")
    print(f"Execution time is: {execution_time:.4f} seconds with {N} points, {D} dimensions, and {K} clusters")
 

def test_knn():
    # test data
    N, D, A, X, K = testdata_knn(args.testfile, 5)  # Testing for 5 query points
    knn_result = our_knn(N, D, A, X, K, process_distance_func(args.dist))
    
    print("KNN (task 1) results are:")
    print(knn_result)

def test_knn_detailed():
    N, D, A, X, K = testdata_knn(args.testfile)
    
    print(f"\nTesting KNN with:")
    print(f"Database size (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Query points (M): {X.shape[0] if isinstance(X, np.ndarray) else 1}")
    print(f"Neighbors (K): {K}")
    
    # Run multiple times for timing statistics
    num_runs = 5
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        knn_result = our_knn(N, D, A, X, K, process_distance_func(args.dist))
        times.append(time.time() - start_time)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"KNN: {mean_time:.4f} ± {std_time:.4f} seconds")
    
    return knn_result
    
def test_ann():
    N, D, A, X, K = testdata_ann(args.testfile, 5)
    ann_result = our_ann_kmeans(N, D, A, X, K, process_distance_func(args.dist))
    print("ANN (task 2.2) results are:")
    print(ann_result)


def test_ann_detailed():
    # test data
    N, D, A, X, K = testdata_ann(args.testfile, M=5)  # Test with 5 query points
    
    print("\n=== Testing Approximate Nearest Neighbors ===")
    print(f"Database size (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Number of queries (M): {X.shape[0]}")
    print(f"Number of neighbors (K): {K}")
    print(f"Distance metric: {args.dist}")
    
    T = 5  # T is the number of trials
    times = []
    results = []
    
    print("\nRunning trials...")
    for t in range(T):
        start_time = time.time()
        ann_result = our_ann_kmeans(N, D, A, X, K, process_distance_func(args.dist))
        trial_time = time.time() - start_time
        times.append(trial_time)
        results.append(ann_result)
        print(f"Trial {t+1}: {trial_time:.4f} seconds")
    
    # analyse results
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print("\n=== Results ===")
    print(f"Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Fastest trial: {min(times):.4f} seconds")
    print(f"Slowest trial: {max(times):.4f} seconds")
    
    # checking for consistency across trials
    if T > 1:
        print("\n=== Consistency Analysis ===")
        base_result = results[0]
        for t in range(1, T):
            match_rate = np.mean([len(set(r1) & set(r2)) / K 
                                for r1, r2 in zip(base_result, results[t])])
            print(f"Trial {t+1} match rate with Trial 1: {match_rate:.4f}")
    
    return results[0]


def recall_rate(list1, list2):
    return len(set(list1) & set(list2)) / len(list1)

def recall_test(knn_function, ann_function, T=10):
    N, D, A, X, K = testdata_ann(args.testfile)
    
    # start
    start_time = time.time()
    knn_results = knn_function(N, D, A, X, K)
    knn_time = time.time() - start_time
    
    total_recall = 0.0
    ann_total_time = 0.0
    
    for _ in range(T):
        start_time = time.time()
        ann_results = ann_function(N, D, A, X, K)
        ann_total_time += time.time() - start_time
        total_recall += recall_rate(knn_results, ann_results)
    
    avg_recall = total_recall/T
    avg_ann_time = ann_total_time/T
    
    print(f"\n=== Recall Test Results ===")
    print(f"Functions tested: {knn_function.__name__} vs {ann_function.__name__}")
    print(f"Parameters: K={K}, T={T}")
    print(f"Exact KNN time: {knn_time:.4f} seconds")
    print(f"Average ANN time: {avg_ann_time:.4f} seconds")
    print(f"Average speedup: {knn_time/avg_ann_time:.2f}x")
    print(f"Average recall: {avg_recall:.4f}")


def init_gpu():
    """Initialize and warm up GPU"""
    try:
        # Get device information
        device = cp.cuda.runtime.getDeviceProperties(0)
        print(f"\nGPU Device: {device['name'].decode()}")
        print(f"Compute Capability: {device['computeMajor']}.{device['computeMinor']}")
        print(f"Memory: {device['totalGlobalMem'] / 1e9:.2f} GB")
        
        # Set memory pool allocator
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        
        # Warm up GPU
        x = cp.arange(1000)
        x = cp.sum(x)  # Force computation
        cp.cuda.Stream.null.synchronize()
        
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        print("No GPU found. Look into your CUDA PATH variables!")
        return False
    

if __name__ == "__main__":

    init_gpu()
    
    print("Starting task 1")
    test_kmeans_detailed()
    print("Starting task 2")
    test_knn_detailed()
    print("Starting task 3")
    test_ann_detailed()

