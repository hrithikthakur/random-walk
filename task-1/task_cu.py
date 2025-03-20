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
    """
    Compute L2 distance between points in X and Y
    Handles 1D arrays properly
    """
    # Ensure Y is 2D if it's a single point
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    
    # Use broadcasting to compute distances
    XX = cp.sum(X ** 2, axis=1, keepdims=True)
    YY = cp.sum(Y ** 2, axis=1, keepdims=True)
    
    # Calculate squared distances
    distances = XX + YY.T - 2.0 * cp.dot(X, Y.T)
    
    # Return sqrt of distances for L2 norm
    return cp.sqrt(cp.maximum(distances, 0.0))

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
        exact_distances = distance_func(X[i:i+1], candidates)
        final_indices[i] = candidate_indices[i][cp.argsort(exact_distances[0])[:K]]
    
    return cp.asnumpy(final_indices)




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

# Running the ANN function
def our_ann(N, D, A, X, K, distance_func):
    """
    Improved ANN implementation with proper index handling
    """
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    M = X_gpu.shape[0]
    
    # Use more clusters than K for better space partitioning
    num_clusters = min(100, int(cp.sqrt(N)))  # Better value than just K
    K1 = min(4, num_clusters)  # Check 4 closest clusters
    K2 = min(1000, N//10)      # Check 1000 candidates per cluster
    
    # Run k-means clustering with proper number of clusters
    centroids = cp.asarray(our_kmeans(N, D, A_gpu, num_clusters, distance_func))
    
    # Find cluster assignments for all points
    all_distances = distance_func(A_gpu, centroids)
    cluster_labels = cp.argmin(all_distances, axis=1)
    
    # Find closest clusters for each query
    query_distances = distance_func(X_gpu, centroids)
    closest_clusters = cp.argsort(query_distances, axis=1)[:, :K1]
    
    results = []
    for i in range(M):
        x = X_gpu[i:i+1]  # Keep as 2D array
        clusters = closest_clusters[i]
        
        # Collect candidates from each cluster
        candidates = []
        for c in clusters:
            # Get points that belong to this cluster
            cluster_indices = cp.where(cluster_labels == c)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Calculate distances to all points in cluster
            cluster_points = A_gpu[cluster_indices]
            dists = distance_func(x, cluster_points)[0]  # Get first row
            
            # Get K2 nearest points in this cluster
            nearest_indices = cp.argsort(dists)[:K2]
            
            # Add original indices to candidates
            candidates.extend(cluster_indices[nearest_indices].get().tolist())
        
        # If we have candidates, find K nearest among them
        if candidates:
            candidates = cp.array(candidates)
            candidate_points = A_gpu[candidates]
            final_dists = distance_func(x, candidate_points)[0]
            final_indices = candidates[cp.argsort(final_dists)[:K]]
            results.append(final_indices)
        else:
            # Fallback to basic search
            dists = distance_func(x, A_gpu)[0]
            results.append(cp.argsort(dists)[:K])
    
    return cp.array(results)



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
    ann_result = our_ann(N, D, A, X, K, process_distance_func(args.dist))
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
        ann_result = our_ann(N, D, A, X, K, process_distance_func(args.dist))
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
            match_rate = np.mean([len(set(r1.tolist()) & set(r2.tolist())) / K for r1, r2 in zip(exact_results.T, ann_results.T)
    ])
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
    print("Inside Main")
    #warm up
    init_gpu()

    
    
    print("Test Kmeans")
    test_kmeans_detailed()
    print("Test KNN")
    test_knn_detailed()
    print("Test ANN KMEANS")
    # test_ann_detailed()
    print("Test ANN LSH")
    # test_ann_lsh_detailed()
    print("Recall test")
    recall_test(our_knn, our_ann)
    recall_test(our_knn, our_ann_lsh)


