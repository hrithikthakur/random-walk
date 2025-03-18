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
        print(f"""WARN: you were trying to use a distance that was not cosine with the LSH implementation. 
              This LSH implementation is built to be used with cosine distance specifically.
              Therefore, intead of using specified {distance_func.__name__} distance, cosine distance is still being used.""")
    
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

def our_ann_efficient(N, D, A, X, K, metric='l2', k1_factor=3.0, k2_factor=6.0):
    """
    Efficient ANN implementation with optimized recall rate
    Assumes GPU is always available
    
    Args:
        N: Number of database points
        D: Dimensions of points
        A: Database points of shape (N, D)
        X: Query points of shape (M, D)
        K: Number of nearest neighbors to find
        metric: Distance metric ('l2' or 'cosine')
        k1_factor: Controls number of clusters (higher = more clusters = higher recall)
        k2_factor: Controls candidates per cluster (higher = more candidates = higher recall)
    
    Returns:
        Array of indices of shape (K, M) where M is number of query points
    """
    # Convert inputs to CuPy arrays
    A = cp.asarray(A)
    X = cp.asarray(X)
    M = X.shape[0]  # Number of query points
    
    # Compute optimal parameters for dataset characteristics
    K1 = max(5, min(int(np.sqrt(N) * k1_factor), N//4))
    K2 = max(K * 2, min(int(K * k2_factor), N//4))
    n_probe = max(2, min(K1//4, 5))  # Fixed probe depth, optimized for recall/performance
    
    print(f"ANN parameters: K1={K1}, K2={K2}, probe={n_probe}")
    
    # Setup distance function and normalize if needed
    if metric == 'cosine':
        distance_fn = lambda x, y: cosine_distance(x, y)
        A = A / (cp.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        X = X / (cp.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    else:  # default to l2
        distance_fn = lambda x, y: cp.linalg.norm(x - y, axis=1)
    
    # Run k-means clustering once with better initialization
    cluster_labels = our_kmeans(N, D, A, K1, metric=metric)
    centroids = cp.zeros((K1, D))
    
    # Compute centroids and analyze cluster sizes
    cluster_sizes = cp.zeros(K1)
    for k in range(K1):
        mask = cluster_labels == k
        size = cp.sum(mask)
        cluster_sizes[k] = size
        if size > 0:
            centroids[k] = cp.mean(A[mask], axis=0)
            if metric == 'cosine':
                centroids[k] = centroids[k] / (cp.linalg.norm(centroids[k]) + 1e-8)
    
    min_size = int(cp.min(cluster_sizes))
    max_size = int(cp.max(cluster_sizes))
    avg_size = float(cp.mean(cluster_sizes))
    print(f"Cluster stats: avg={avg_size:.1f}, min={min_size}, max={max_size}")
    
    # Process queries in batch for efficiency
    results = []
    for x in X:
        # Find closest clusters
        distances_to_centroids = distance_fn(centroids, x)
        closest_clusters = cp.argsort(distances_to_centroids)[:n_probe]
        
        # Gather candidates from clusters
        candidates = []
        for cluster_idx in closest_clusters:
            cluster_indices = cp.where(cluster_labels == cluster_idx)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Select candidates from this cluster
            cluster_points = A[cluster_indices]
            cluster_distances = distance_fn(cluster_points, x)
            
            # Get more candidates from larger clusters
            size_factor = min(3.0, cluster_sizes[cluster_idx] / avg_size)
            local_k2 = min(int(K2 * size_factor), len(cluster_indices))
            
            nearest_in_cluster = cp.argsort(cluster_distances)[:local_k2]
            candidates.extend(cluster_indices[nearest_in_cluster].get().tolist())
        
        # Fallback if we don't have enough candidates
        if len(candidates) < K:
            # Add points from other clusters
            remaining = [c for c in range(K1) if c not in closest_clusters]
            for c in remaining[:3]:  # Check up to 3 more clusters
                cluster_indices = cp.where(cluster_labels == c)[0]
                if len(cluster_indices) > 0:
                    candidates.extend(cluster_indices.get().tolist()[:K])
                if len(candidates) >= K * 3:
                    break
        
        # Final refinement
        if candidates:
            candidates = cp.array(candidates)
            candidate_points = A[candidates]
            final_distances = distance_fn(candidate_points, x)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            # Fall back to full search
            distances = distance_fn(A, x)
            indices = cp.argsort(distances)[:K]
            results.append(indices.get())
    
    return cp.array(results).T

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
    N, D, A, X, K = testdata_knn(args.testfile, 1)  # Testing for 1 query points
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
    N, D, A, X, K = testdata_ann(args.testfile, 1)
    ann_result = our_ann_efficient(N, D, A, X, K, process_distance_func(args.dist))
    print("ANN (task 2.2) results are:")
    print(ann_result)


def test_ann_detailed():
    # test data
    N, D, A, X, K = testdata_ann(args.testfile, M=1)  # Test with 1 query points
    
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
        ann_result = our_ann_efficient(N, D, A, X, K, process_distance_func(args.dist))
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

def test_ann_lsh():
    N, D, A, X, K = testdata_ann(args.testfile, 1)
    ann_result = our_ann_lsh(N, D, A, X, K, process_distance_func(args.dist))
    print("ANN (task 2.2) results are:")
    print(ann_result)

def test_ann_lsh_detailed():
    # test data
    N, D, A, X, K = testdata_ann(args.testfile, M=1)  # Test with 1 query points
    
    print("\n=== Testing LSH Approximate Nearest Neighbors ===")
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
        ann_result = our_ann_lsh(N, D, A, X, K, process_distance_func(args.dist))
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

def recall_rate(exact_neighbors, approx_neighbors):
    # convert to numpy arrays from cupy arrays if needed
    if isinstance(exact_neighbors, cp.ndarray):
        exact_neighbors = cp.asnumpy(exact_neighbors)
    if isinstance(approx_neighbors, cp.ndarray):
        approx_neighbors = cp.asnumpy(approx_neighbors)
    
    # convret to flat lists
    exact_list = exact_neighbors.flatten().tolist()
    approx_list = approx_neighbors.flatten().tolist()
    
    # calculate recall rate
    correct_matches = len(set(exact_list) & set(approx_list))
    return correct_matches / len(exact_list)

def recall_test(knn_function, ann_function, T=10):
    N, D, A, X, K = testdata_ann(args.testfile)
    
    print("\nRunning recall test...")
    print(f"N={N}, D={D}, K={K}")
    
    # knn
    print("Computing exact KNN...")
    knn_results = knn_function(N, D, A, X, K, process_distance_func(args.dist))
    
    # Run ANN multiple times
    print(f"Computing ANN {T} times...")
    total_recall = 0.0
    recalls = []
    
    for t in range(T):
        ann_results = ann_function(N, D, A, X, K, process_distance_func(args.dist))
        recall = recall_rate(knn_results, ann_results)
        recalls.append(recall)
        total_recall += recall
        print(f"Trial {t+1}: Recall = {recall:.4f}")
    
    avg_recall = total_recall / T
    std_recall = np.std(recalls)
    
    print(f"\nFinal Results:")
    print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Best Recall:    {max(recalls):.4f}")
    print(f"Worst Recall:   {min(recalls):.4f}")


def init_gpu():
    try:
        device = cp.cuda.runtime.getDeviceProperties(0)
        print(f"\nGPU Device: {device['name'].decode()}")
        print(f"Total Memory: {device['totalGlobalMem'] / 1e9:.2f} GB")
        
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        
        x = cp.arange(1000)
        x = cp.sum(x)
        cp.cuda.Stream.null.synchronize()
        
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        print("CUDA path variables might not have been set correctly!")
        return False
    
if __name__ == "__main__":

    #warm up
    init_gpu()
    
    print("Test Kmeans")
    test_kmeans_detailed()
    print("Test KNN")
    test_knn_detailed()
    print("Test ANN KMEANS")
    test_ann_detailed()
    print("Test ANN LSH")
    test_ann_lsh_detailed()
    print("Recall test")
    recall_test(our_knn, our_ann_efficient)
    recall_test(our_knn, our_ann_lsh)


