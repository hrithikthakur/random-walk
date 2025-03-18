import numpy as np
import time
import argparse
import os
import glob

# Try to import CuPy, but provide fallback if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as cp  # Fallback to NumPy with same alias
    CUPY_AVAILABLE = False

# Global flag to track if GPU is working
GPU_AVAILABLE = False

def init_gpu():
    """Initialize GPU and set up memory pool, with fallback"""
    global GPU_AVAILABLE
    
    # Check if CuPy is available at all
    if not CUPY_AVAILABLE:
        print("CuPy not installed. Using CPU (NumPy) instead.")
        return False
        
    try:
        # Try to initialize GPU
        device_id = cp.cuda.runtime.getDevice()
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        print(f"Using device: cuda")
        print(f"Device {device_id}: {device_props['name'].decode()}")
        print(f"Compute capability: {device_props['major']}.{device_props['minor']}")
        
        # Set memory pool allocator
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        
        GPU_AVAILABLE = True
        return True
    except Exception as e:
        print(f"No GPU found or error initializing: {e}")
        print("Falling back to CPU (NumPy) computations")
        
        # Handle the CuPy to NumPy transition
        global cp
        import numpy as np
        cp = np  # Replace cp with np for all future calls
        
        GPU_AVAILABLE = False
        return False

# Utility functions for array conversion
def to_numpy(arr):
    """Convert array to NumPy if it's CuPy"""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray) and hasattr(arr, 'get'):
        return arr.get()
    return arr

def array_equal(a, b):
    """Compare arrays regardless of whether they're CuPy or NumPy"""
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    return np.array_equal(a_np, b_np)

def asarray(arr):
    """Convert to either CuPy or NumPy array depending on GPU availability"""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)

def cosine_distance(A, B):
    """Compute cosine distance between points in A and B, works with either CuPy or NumPy"""
    # Normalize vectors
    A_norm = A / (cp.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    if A is B:
        B_norm = A_norm
    else:
        B_norm = B / (cp.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity and convert to distance
    if len(B_norm.shape) == 1:
        B_norm = B_norm.reshape(1, -1)
    similarity = cp.dot(A_norm, B_norm.T)
    return 1 - similarity

def our_kmeans(N, D, A, K, metric='l2', max_iter=100):
    """K-means clustering with either CuPy or NumPy"""
    # Convert input to appropriate array type
    A = asarray(A)
    
    # Initialize centroids by selecting K random data points
    idx = cp.random.choice(N, K, replace=False)
    centroids = A[idx].copy()
    
    # Distance function based on metric
    if metric == 'cosine':
        distance_fn = lambda x, y: cosine_distance(x, y.reshape(1, -1)).ravel()
    else:  # default to l2
        distance_fn = lambda x, y: cp.linalg.norm(x - y, axis=1)
    
    # Iterative refinement
    for _ in range(max_iter):
        # Assign each point to nearest centroid
        distances = cp.zeros((N, K))
        for k in range(K):
            distances[:, k] = distance_fn(A, centroids[k])
        labels = cp.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = cp.zeros_like(centroids)
        for k in range(K):
            points = A[labels == k]
            if len(points) > 0:
                new_centroids[k] = cp.mean(points, axis=0)
                if metric == 'cosine':
                    # Normalize for cosine distance
                    new_centroids[k] = new_centroids[k] / (cp.linalg.norm(new_centroids[k]) + 1e-8)
            else:
                # If a cluster is empty, reinitialize with a random point
                new_centroids[k] = A[cp.random.randint(0, N)]
        
        # Check for convergence
        if array_equal(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels

def our_knn(N, D, A, X, K, metric='l2'):
    """K-nearest neighbors with either CuPy or NumPy"""
    # Convert inputs to appropriate array type
    A = asarray(A)
    X = asarray(X)
    
    results = []
    
    # Distance function based on metric
    if metric == 'cosine':
        for x in X:
            distances = cosine_distance(A, x)
            indices = cp.argsort(distances)[:K]
            results.append(indices)
    else:  # default to l2
        for x in X:
            distances = cp.linalg.norm(A - x, axis=1)
            indices = cp.argsort(distances)[:K]
            results.append(indices)
    
    return cp.array(results).T

def our_ann(N, D, A, X, K, metric='l2', k1_factor=3.0, k2_factor=5.0, num_probe=3, ensure_recall=0.8):
    """
    Improved ANN implementation with better recall rate and distance metric options
    Works with either CuPy or NumPy
    
    Args:
        N: Number of database points
        D: Dimensions of points
        A: Database points of shape (N, D)
        X: Query points of shape (M, D)
        K: Number of nearest neighbors to find
        metric: Distance metric ('l2' or 'cosine')
        k1_factor: Controls number of clusters (higher = more clusters = higher recall but slower)
        k2_factor: Controls candidates per cluster (higher = more candidates = higher recall but slower)
        num_probe: Number of closest clusters to search
        ensure_recall: Target minimum recall rate (0-1.0, higher values enable more exhaustive search)
    
    Returns:
        Array of indices of shape (K, M) where M is number of query points
    """
    # Convert inputs to appropriate array type
    A = asarray(A)
    X = asarray(X)
    M = X.shape[0]  # Number of query points
    
    # Adaptive hyperparameters based on dataset characteristics
    density_factor = min(1.0, 10000 / N)  # Adjust based on dataset size
    dim_factor = min(1.0, 100 / D)        # Adjust based on dimensionality
    
    # Apply factors with user control
    K1 = max(5, min(int(np.sqrt(N) * k1_factor * density_factor), N//3))
    K2 = max(K * 2, min(int(K * k2_factor * dim_factor), N//3))
    num_probe = max(1, min(num_probe, K1-1))
    
    print(f"ANN parameters - Clusters (K1): {K1}, Candidates per cluster (K2): {K2}, Probe depth: {num_probe}")
    
    # Distance function based on metric
    if metric == 'cosine':
        distance_fn = lambda x, y: cosine_distance(x, y)
        # Normalize vectors for cosine similarity
        A = A / (cp.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        X = X / (cp.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    else:  # default to l2
        distance_fn = lambda x, y: cp.linalg.norm(x - y, axis=1)
    
    # Step 1: Better Clustering with multiple restarts
    best_labels = None
    best_centroids = None
    best_distortion = float('inf')
    
    print("Running k-means clustering...")
    for restart in range(3):  # Multiple restarts to find better clustering
        cluster_labels = our_kmeans(N, D, A, K1, metric=metric)
        centroids = cp.zeros((K1, D))
        
        # Compute centroids
        for k in range(K1):
            mask = cluster_labels == k
            if cp.any(mask):
                centroids[k] = cp.mean(A[mask], axis=0)
                if metric == 'cosine':
                    centroids[k] = centroids[k] / (cp.linalg.norm(centroids[k]) + 1e-8)
        
        # Calculate distortion
        distortion = 0
        for k in range(K1):
            mask = cluster_labels == k
            if cp.any(mask):
                cluster_points = A[mask]
                if metric == 'cosine':
                    dist = cp.sum(1 - cp.dot(cluster_points, centroids[k]))
                else:
                    dist = cp.sum(cp.sum((cluster_points - centroids[k])**2, axis=1))
                distortion += float(dist)
        
        if distortion < best_distortion:
            best_distortion = distortion
            best_labels = cluster_labels
            best_centroids = centroids
    
    cluster_labels = best_labels
    centroids = best_centroids
    
    # Analyze cluster sizes for balanced probing
    cluster_sizes = cp.array([cp.sum(cluster_labels == k) for k in range(K1)])
    if GPU_AVAILABLE:
        cluster_sizes_np = cluster_sizes.get()
    else:
        cluster_sizes_np = cluster_sizes
    avg_size = float(cp.mean(cluster_sizes))
    
    print(f"Cluster stats - Avg size: {avg_size:.1f}, Min: {cp.min(cluster_sizes)}, Max: {cp.max(cluster_sizes)}")
    
    # Step 2: Process each query point
    results = []
    
    # Track recall quality
    if ensure_recall > 0:
        # Get exact results for a small subset to adapt parameters
        sample_size = min(3, M)
        print(f"Computing ground truth for {sample_size} sample queries to validate recall...")
        validation_indices = cp.arange(min(sample_size, M))
        exact_results = []
        for x_idx in validation_indices:
            distances = distance_fn(A, X[x_idx])
            exact_indices = cp.argsort(distances)[:K]
            if GPU_AVAILABLE:
                exact_results.append(exact_indices.get())
            else:
                exact_results.append(exact_indices)
    
    print(f"Processing {M} query points...")
    for x_idx, x in enumerate(X):
        # Find distances to all centroids using selected metric
        if metric == 'cosine':
            centroid_distances = cosine_distance(centroids, x).ravel()
        else:
            centroid_distances = cp.linalg.norm(centroids - x, axis=1)
        
        # Weight clusters by size (prefer searching smaller clusters first)
        adjusted_distances = centroid_distances * (cluster_sizes / avg_size) ** 0.3
        
        # Find nearest clusters with sufficient points
        nearest_clusters = []
        sorted_clusters = cp.argsort(adjusted_distances)
        
        # Select clusters to probe
        for c in sorted_clusters:
            if len(nearest_clusters) >= num_probe:
                break
            if cluster_sizes[c] >= K:
                nearest_clusters.append(c)
        
        # Step 3: Gather candidates with dynamic K2
        candidates = []
        for cluster_idx in nearest_clusters:
            cluster_points = A[cluster_labels == cluster_idx]
            cluster_indices = cp.where(cluster_labels == cluster_idx)[0]
            
            if len(cluster_points) == 0:
                continue
            
            # Calculate distances using selected metric
            distances = distance_fn(cluster_points, x)
            
            # Adaptively select more candidates from large clusters
            size_ratio = float(cluster_sizes[cluster_idx]) / avg_size
            local_K2 = min(int(K2 * size_ratio), len(cluster_points))
            local_K2 = max(K, local_K2)  # Ensure we get at least K candidates
            
            nearest = cp.argsort(distances)[:local_K2]
            if GPU_AVAILABLE:
                candidates.extend(cluster_indices[nearest].get().tolist())
            else:
                candidates.extend(cluster_indices[nearest].tolist())
        
        # Step 4: Enhanced refinement
        if len(candidates) < K * 2:
            # If we don't have enough candidates, probe more clusters
            remaining_clusters = [c for c in cp.argsort(centroid_distances) 
                               if c not in nearest_clusters]
            
            for cluster_idx in remaining_clusters:
                if len(candidates) >= K * 2:
                    break
                cluster_points = A[cluster_labels == cluster_idx]
                cluster_indices = cp.where(cluster_labels == cluster_idx)[0]
                
                if len(cluster_points) == 0:
                    continue
                
                distances = distance_fn(cluster_points, x)
                more_indices = cp.argsort(distances)[:K]
                if GPU_AVAILABLE:
                    candidates.extend(cluster_indices[more_indices].get().tolist())
                else:
                    candidates.extend(cluster_indices[more_indices].tolist())
        
        # If using recall validation, check and potentially boost search
        if ensure_recall > 0 and x_idx in validation_indices:
            validation_idx = int(cp.where(validation_indices == x_idx)[0])
            
            # Keep expanding search until we reach target recall
            current_recall = 0
            probe_expansion = 1
            exact_indices = exact_results[validation_idx]
            
            while current_recall < ensure_recall and probe_expansion <= 5:
                # Final selection using selected metric
                if candidates:
                    candidates = cp.array(candidates)
                    candidate_points = A[candidates]
                    final_distances = distance_fn(candidate_points, x)
                    final_indices = candidates[cp.argsort(final_distances)[:K]]
                    
                    if GPU_AVAILABLE:
                        result = final_indices.get()
                    else:
                        result = final_indices
                    
                    # Calculate recall
                    intersection = len(set(result.flatten()) & set(exact_indices.flatten()))
                    current_recall = intersection / K
                    
                    if current_recall >= ensure_recall:
                        break
                    
                # If recall is too low, add more candidates
                probe_expansion += 1
                extra_clusters = sorted_clusters[num_probe:num_probe+probe_expansion]
                for c in extra_clusters:
                    cluster_points = A[cluster_labels == c]
                    cluster_indices = cp.where(cluster_labels == c)[0]
                    
                    if len(cluster_points) == 0:
                        continue
                    
                    distances = distance_fn(cluster_points, x)
                    more_indices = cp.argsort(distances)[:K2]
                    if GPU_AVAILABLE:
                        candidates.extend(cluster_indices[more_indices].get().tolist())
                    else:
                        candidates.extend(cluster_indices[more_indices].tolist())
                
                # De-duplicate candidates
                candidates = list(set(candidates))
            
            print(f"Sample query {validation_idx}: Achieved recall {current_recall:.2f} with {probe_expansion} expansion steps")
            
            # Update parameters based on validation results
            if current_recall < ensure_recall:
                # Boost parameters for remaining queries
                num_probe = min(num_probe + 1, K1 - 1)
                K2 = int(K2 * 1.5)
                print(f"Boosting search parameters - New probe depth: {num_probe}, candidates: {K2}")
        
        # Final selection using appropriate metric
        if candidates:
            candidates = cp.array(candidates)
            candidate_points = A[candidates]
            final_distances = distance_fn(candidate_points, x)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            
            if GPU_AVAILABLE:
                results.append(final_indices.get())
            else:
                results.append(final_indices)
        else:
            # Fallback to brute force for this query
            distances = distance_fn(A, x)
            indices = cp.argsort(distances)[:K]
            
            if GPU_AVAILABLE:
                results.append(indices.get())
            else:
                results.append(indices)
    
    return cp.array(results).T

def recall_rate(exact_nn, approx_nn):
    """Calculate recall rate between exact and approximate nearest neighbors"""
    # Convert to numpy and flatten
    if GPU_AVAILABLE and isinstance(exact_nn, cp.ndarray) and hasattr(exact_nn, 'get'):
        exact_nn = exact_nn.get()
    if GPU_AVAILABLE and isinstance(approx_nn, cp.ndarray) and hasattr(approx_nn, 'get'):
        approx_nn = approx_nn.get()
    
    # Ensure both are flattened
    exact_nn = np.array(exact_nn).flatten()
    approx_nn = np.array(approx_nn).flatten()
    
    # Calculate intersection size
    intersection = len(set(exact_nn).intersection(set(approx_nn)))
    recall = intersection / len(exact_nn)
    return recall

def process_distance_func(dist_name):
    """Convert distance function name to internal format"""
    if dist_name == 'cosine':
        return 'cosine'
    else:
        return 'l2'  # default

def testdata_ann(testfile):
    """Load ANN test data"""
    print(f"Loading ANN test data from {testfile}")
    data = np.load(testfile)
    N = data['N']
    D = data['D']
    A = data['A']
    X = data['X']
    K = data['K']
    return N, D, A, X, K

def test_kmeans_detailed():
    """Run detailed K-means test with specified test file"""
    N, D, A, K = np.random.randint(800, 1200), 100, np.random.random((1000, 100)), 10
    
    if args.testfile:
        data = np.load(args.testfile)
        N = data['N']
        D = data['D']
        A = data['A']
        K = min(int(np.sqrt(N)), 20)  # reasonable K value
    
    metric = process_distance_func(args.dist) if hasattr(args, 'dist') else 'l2'
    
    print("\nK-Means Clustering Test:")
    print(f"Number of points (N): {N}")
    print(f"Dimensions (D): {D}")
    print(f"Number of clusters (K): {K}")
    print(f"Data shape: {A.shape}")
    print(f"Distance metric: {metric}")
    
    # Time the execution
    start_time = time.time()
    kmeans_result = our_kmeans(N, D, A, K, metric)
    elapsed = time.time() - start_time
    
    # Verify results and provide information
    if kmeans_result is not None:
        print(f"K-means execution time: {elapsed:.4f} seconds")
        
        # Convert to numpy for analysis
        result_np = to_numpy(kmeans_result)
        
        # Analyze cluster distribution
        unique_labels, counts = np.unique(result_np, return_counts=True)
        print(f"Number of clusters formed: {len(unique_labels)}")
        print(f"Average cluster size: {np.mean(counts):.1f} points")
        print(f"Largest cluster: {np.max(counts)} points")
        print(f"Smallest cluster: {np.min(counts)} points")
    else:
        print("K-means failed to produce results")

def test_ann_methods():
    """Test and compare different ANN methods with different distance metrics"""
    print("\nTesting ANN methods...")
    
    try:
        N, D, A, X, K = testdata_ann(args.testfile)
    except:
        print("Error loading test data, using random data")
        N, D = 1000, 50
        A = np.random.random((N, D))
        X = np.random.random((5, D))
        K = 10
    
    # Get the distance metric from args
    metric = process_distance_func(args.dist) if hasattr(args, 'dist') else 'l2'
    
    # Get parameters from args or use defaults
    k1_factor = float(args.k1) if hasattr(args, 'k1') else 3.0
    k2_factor = float(args.k2) if hasattr(args, 'k2') else 5.0
    num_probe = int(args.probe) if hasattr(args, 'probe') else 3
    ensure_recall = float(args.recall) if hasattr(args, 'recall') else 0.0
    
    print(f"\nTesting with {metric} distance:")
    print(f"Dataset: N={N}, D={D}, Query points={X.shape[0]}, K={K}")
    print(f"Parameters: k1_factor={k1_factor}, k2_factor={k2_factor}, probe={num_probe}, target_recall={ensure_recall}")
    
    # Get ground truth
    print("Computing exact KNN (ground truth)...")
    start_time = time.time()
    exact_results = our_knn(N, D, A, X, K, metric=metric)
    exact_time = time.time() - start_time
    print(f"Exact KNN time: {exact_time:.4f} seconds")
    
    # Test ANN implementation
    print("\nTesting our ANN implementation...")
    start_time = time.time()
    ann_results = our_ann(N, D, A, X, K, metric=metric, 
                         k1_factor=k1_factor, k2_factor=k2_factor, 
                         num_probe=num_probe, ensure_recall=ensure_recall)
    ann_time = time.time() - start_time
    
    recalls = []
    for i in range(X.shape[0]):
        recall = recall_rate(exact_results[:, i], ann_results[:, i])
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    speedup = exact_time / ann_time if ann_time > 0 else 0
    
    print(f"\nResults Summary:")
    print(f"ANN time: {ann_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Recall ({metric}): {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Min recall: {min(recalls):.4f}, Max recall: {max(recalls):.4f}")

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

def test_ann_efficient():
    """Test the efficient ANN implementation"""
    print("\nTesting Efficient ANN...")
    
    try:
        N, D, A, X, K = testdata_ann(args.testfile)
    except:
        print("Error loading test data, using random data")
        N, D = 1000, 50
        A = np.random.random((N, D))
        X = np.random.random((5, D))
        K = 10
    
    # Get parameters
    metric = process_distance_func(args.dist) if hasattr(args, 'dist') else 'l2'
    k1_factor = float(args.k1) if hasattr(args, 'k1') else 3.0
    k2_factor = float(args.k2) if hasattr(args, 'k2') else 6.0
    
    print(f"\nDataset: N={N}, D={D}, Queries={X.shape[0]}, K={K}")
    print(f"Using {metric} distance, k1={k1_factor}, k2={k2_factor}")
    
    # Get ground truth
    print("Computing exact KNN...")
    start_time = time.time()
    exact_results = our_knn(N, D, A, X, K, metric=metric)
    exact_time = time.time() - start_time
    print(f"Exact KNN time: {exact_time:.4f}s")
    
    # Test efficient ANN
    print("\nRunning efficient ANN...")
    start_time = time.time()
    ann_results = our_ann_efficient(N, D, A, X, K, metric=metric, k1_factor=k1_factor, k2_factor=k2_factor)
    ann_time = time.time() - start_time
    
    # Evaluate results
    recalls = []
    for i in range(X.shape[0]):
        # Convert arrays to NumPy for recall calculation
        exact_np = exact_results[:, i].get()
        ann_np = ann_results[:, i].get()
        recall = recall_rate(exact_np, ann_np)
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    speedup = exact_time / ann_time if ann_time > 0 else 0
    
    print(f"\nResults:")
    print(f"ANN time: {ann_time:.4f}s (speedup: {speedup:.2f}x)")
    print(f"Average recall: {mean_recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN and ANN Testing')
    parser.add_argument('--testfile', type=str, default=None, help='Test data file to use')
    parser.add_argument('--dist', type=str, default='l2', help='Distance metric to use: l2, cosine')
    parser.add_argument('--test', type=str, default='kmeans', help='Test to run: kmeans, ann, efficient')
    parser.add_argument('--k1', type=float, default=3.0, help='K1 factor for clusters in ANN')
    parser.add_argument('--k2', type=float, default=6.0, help='K2 factor for candidates per cluster in ANN')
    args = parser.parse_args()
    
    # Initialize GPU or fall back to CPU
    init_gpu()
    
    if args.test == 'kmeans':
        print("\nTest Kmeans")
        test_kmeans_detailed()
    elif args.test == 'ann':
        print("\nTest ANN")
        test_ann_methods()
    elif args.test == 'efficient':
        print("\nTest Efficient ANN")
        test_ann_efficient()
    else:
        print(f"Unknown test: {args.test}")
        print("Available tests: kmeans, ann, efficient") 