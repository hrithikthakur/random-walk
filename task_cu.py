def cosine_distance(A, B):
    """Compute cosine distance between points in A and B"""
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

def our_ann(N, D, A, X, K, metric='l2'):
    """
    Improved ANN implementation with better recall rate and distance metric options
    Args:
        metric: 'l2' for Euclidean distance or 'cosine' for cosine distance
    """
    # Hyperparameters tuned for better recall
    K1 = min(int(np.sqrt(N)) * 3, N//5)    
    K2 = min(K * 5, N//5)                  
    num_probe = 3                          
    
    # Ensure inputs are on GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    # Distance function based on metric
    if metric == 'cosine':
        distance_fn = lambda x, y: cosine_distance(x, y)
    else:  # default to l2
        distance_fn = lambda x, y: cp.linalg.norm(x - y, axis=1)
    
    # Step 1: Better Clustering
    best_labels = None
    best_centroids = None
    best_distortion = float('inf')
    
    for _ in range(3):
        cluster_labels = our_kmeans(N, D, A_gpu, K1, metric=metric)
        centroids = cp.zeros((K1, D))
        
        for k in range(K1):
            mask = cluster_labels == k
            if cp.any(mask):
                centroids[k] = cp.mean(A_gpu[mask], axis=0)
        
        # Calculate distortion using selected metric
        distortion = 0
        for k in range(K1):
            mask = cluster_labels == k
            if cp.any(mask):
                cluster_points = A_gpu[mask]
                if metric == 'cosine':
                    distances = cp.sum(cosine_distance(cluster_points, centroids[k]))
                else:
                    distances = cp.sum((cluster_points - centroids[k]) ** 2)
                distortion += distances.get()
        
        if distortion < best_distortion:
            best_distortion = distortion
            best_labels = cluster_labels
            best_centroids = centroids
    
    cluster_labels = best_labels
    centroids = best_centroids
    
    # Step 2: Process each query point
    results = []
    for x in X_gpu:
        # Find distances to all centroids using selected metric
        if metric == 'cosine':
            centroid_distances = cosine_distance(centroids, x).ravel()
        else:
            centroid_distances = cp.linalg.norm(centroids - x, axis=1)
        
        cluster_sizes = cp.array([cp.sum(cluster_labels == k) for k in range(K1)])
        nearest_clusters = []
        
        sorted_clusters = cp.argsort(centroid_distances)
        for c in sorted_clusters:
            if len(nearest_clusters) >= num_probe:
                break
            if cluster_sizes[c] >= K:
                nearest_clusters.append(c)
        
        # Step 3: Gather candidates with dynamic K2
        candidates = []
        for cluster_idx in nearest_clusters:
            cluster_points = A_gpu[cluster_labels == cluster_idx]
            cluster_indices = cp.where(cluster_labels == cluster_idx)[0]
            
            if len(cluster_points) == 0:
                continue
            
            # Calculate distances using selected metric
            distances = distance_fn(cluster_points, x)
            
            local_K2 = min(K2, len(cluster_points))
            nearest = cp.argsort(distances)[:local_K2]
            candidates.extend(cluster_indices[nearest].get().tolist())
        
        # Step 4: Enhanced refinement
        if len(candidates) < K * 2:
            remaining_clusters = [c for c in cp.argsort(centroid_distances) 
                               if c not in nearest_clusters]
            
            for cluster_idx in remaining_clusters:
                if len(candidates) >= K * 3:
                    break
                cluster_indices = cp.where(cluster_labels == cluster_idx)[0]
                candidates.extend(cluster_indices.get().tolist()[:K])
        
        # Final selection using selected metric
        if candidates:
            candidates = cp.array(candidates)
            candidate_points = A_gpu[candidates]
            final_distances = distance_fn(candidate_points, x)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            distances = distance_fn(A_gpu, x)
            results.append(cp.argsort(distances)[:K].get())
    
    return cp.array(results).T

def our_ann_basic(N, D, A, X, K, metric='l2'):
    """Basic ANN using random projections"""
    # Convert to GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    # Distance function based on metric
    if metric == 'cosine':
        distance_fn = lambda x, y: cosine_distance(x, y)
        # Normalize vectors for cosine similarity
        A_gpu = A_gpu / (cp.linalg.norm(A_gpu, axis=1, keepdims=True) + 1e-8)
        X_gpu = X_gpu / (cp.linalg.norm(X_gpu, axis=1, keepdims=True) + 1e-8)
    else:  # default to l2
        distance_fn = lambda x, y: cp.linalg.norm(x - y, axis=1)
    
    # Generate random projection matrix
    num_projections = min(D, 32)  # Reduce dimensionality
    P = cp.random.randn(D, num_projections)
    
    # Project data to lower dimension
    A_proj = cp.dot(A_gpu, P)
    X_proj = cp.dot(X_gpu, P)
    
    results = []
    for x in X_proj:
        # Find approximate distances in projected space
        distances = cp.linalg.norm(A_proj - x, axis=1)
        # Get more candidates than needed
        candidate_indices = cp.argsort(distances)[:K*2]
        
        # Refine in original space using appropriate metric
        candidates = A_gpu[candidate_indices]
        true_distances = distance_fn(candidates, X_gpu[0])
        final_indices = candidate_indices[cp.argsort(true_distances)[:K]]
        results.append(final_indices.get())
    
    return cp.array(results).T

def our_ann_kmeans(N, D, A, X, K, metric='l2'):
    """K-means based ANN with improved recall"""
    # Hyperparameters
    K1 = min(int(np.sqrt(N)) * 3, N//5)    # Number of clusters
    K2 = min(K * 5, N//5)                  # Candidates per cluster
    num_probe = 3                          # Number of clusters to probe
    
    # Convert to GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    # Distance function based on metric
    if metric == 'cosine':
        distance_fn = lambda x, y: cosine_distance(x, y)
        # Normalize vectors for cosine similarity
        A_gpu = A_gpu / (cp.linalg.norm(A_gpu, axis=1, keepdims=True) + 1e-8)
        X_gpu = X_gpu / (cp.linalg.norm(X_gpu, axis=1, keepdims=True) + 1e-8)
    else:  # default to l2
        distance_fn = lambda x, y: cp.linalg.norm(x - y, axis=1)
    
    # Clustering with multiple restarts
    best_labels = None
    best_centroids = None
    best_distortion = float('inf')
    
    for _ in range(3):
        cluster_labels = our_kmeans(N, D, A_gpu, K1, metric=metric)
        centroids = cp.zeros((K1, D))
        
        for k in range(K1):
            mask = cluster_labels == k
            if cp.any(mask):
                centroids[k] = cp.mean(A_gpu[mask], axis=0)
                if metric == 'cosine':
                    centroids[k] = centroids[k] / (cp.linalg.norm(centroids[k]) + 1e-8)
        
        # Calculate distortion using selected metric
        distortion = 0
        for k in range(K1):
            mask = cluster_labels == k
            if cp.any(mask):
                cluster_points = A_gpu[mask]
                distances = cp.sum(distance_fn(cluster_points, centroids[k]))
                distortion += distances.get()
        
        if distortion < best_distortion:
            best_distortion = distortion
            best_labels = cluster_labels
            best_centroids = centroids
    
    # Search using best clustering
    results = []
    for x in X_gpu:
        candidates = []
        centroid_distances = distance_fn(best_centroids, x)
        nearest_clusters = cp.argsort(centroid_distances)[:num_probe]
        
        for cluster_idx in nearest_clusters:
            cluster_points = A_gpu[best_labels == cluster_idx]
            cluster_indices = cp.where(best_labels == cluster_idx)[0]
            if len(cluster_points) == 0:
                continue
            
            distances = distance_fn(cluster_points, x)
            nearest = cp.argsort(distances)[:K2]
            candidates.extend(cluster_indices[nearest].get().tolist())
        
        # Final refinement using appropriate metric
        if candidates:
            candidates = cp.array(candidates)
            final_distances = distance_fn(A_gpu[candidates], x)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            distances = distance_fn(A_gpu, x)
            results.append(cp.argsort(distances)[:K].get())
    
    return cp.array(results).T

def our_ann_lsh(N, D, A, X, K, metric='l2'):
    """LSH-based ANN using either random hyperplanes (cosine) or random projections (L2)"""
    # LSH parameters
    num_tables = 8          # Number of hash tables
    num_bits = 16          # Number of bits per hash
    
    # Convert to GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    if metric == 'cosine':
        # For cosine similarity, normalize vectors first
        A_norm = A_gpu / (cp.linalg.norm(A_gpu, axis=1, keepdims=True) + 1e-8)
        X_norm = X_gpu / (cp.linalg.norm(X_gpu, axis=1, keepdims=True) + 1e-8)
        
        # Generate random unit vectors for SimHash
        hyperplanes = cp.random.randn(num_tables, num_bits, D)
        hyperplanes = hyperplanes / cp.linalg.norm(hyperplanes, axis=2, keepdims=True)
        
        # Hash database points using SimHash
        A_hashes = []
        for i in range(num_tables):
            # SimHash: sign(dot product with random unit vectors)
            projections = cp.dot(A_norm, hyperplanes[i].T)
            hash_bits = (projections > 0).astype(cp.int32)
            hash_values = cp.packbits(hash_bits, axis=1)
            A_hashes.append(hash_values)
    else:
        # For L2 distance, use random projections
        # Scale factor for better L2 distance preservation
        scale = cp.sqrt(D)
        hyperplanes = cp.random.randn(num_tables, num_bits, D) / scale
        
        # Hash database points using L2LSH
        A_hashes = []
        for i in range(num_tables):
            projections = cp.dot(A_gpu, hyperplanes[i].T)
            # Quantize projections for L2LSH
            hash_bits = ((projections + 0.5) > 0).astype(cp.int32)
            hash_values = cp.packbits(hash_bits, axis=1)
            A_hashes.append(hash_values)
    
    results = []
    for x_idx, x in enumerate(X_gpu):
        candidates = set()
        
        # Normalize query point for cosine similarity
        if metric == 'cosine':
            x = x / (cp.linalg.norm(x) + 1e-8)
        
        # Hash query point
        for i in range(num_tables):
            proj = cp.dot(x, hyperplanes[i].T)
            if metric == 'cosine':
                hash_bits = (proj > 0).astype(cp.int32)
            else:
                hash_bits = ((proj + 0.5) > 0).astype(cp.int32)
            x_hash = cp.packbits(hash_bits)
            
            # Find points with matching hashes
            matches = cp.where(A_hashes[i] == x_hash)[0]
            candidates.update(matches.get().tolist())
        
        # If too few candidates, add more from other hash buckets
        if len(candidates) < K * 2:
            for i in range(num_tables):
                proj = cp.dot(x, hyperplanes[i].T)
                if metric == 'cosine':
                    hash_bits = (proj > 0).astype(cp.int32)
                else:
                    hash_bits = ((proj + 0.5) > 0).astype(cp.int32)
                x_hash = cp.packbits(hash_bits)
                
                # Find points with similar hashes (Hamming distance 1)
                for bit in range(num_bits):
                    altered_hash = x_hash ^ (1 << bit)
                    matches = cp.where(A_hashes[i] == altered_hash)[0]
                    candidates.update(matches.get().tolist())
                if len(candidates) >= K * 3:
                    break
        
        # Final refinement using the appropriate distance metric
        if candidates:
            candidates = cp.array(list(candidates))
            if metric == 'cosine':
                final_distances = cosine_distance(A_gpu[candidates], x)
            else:
                final_distances = cp.linalg.norm(A_gpu[candidates] - x, axis=1)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            # Fallback to full search with appropriate metric
            if metric == 'cosine':
                distances = cosine_distance(A_gpu, x)
            else:
                distances = cp.linalg.norm(A_gpu - x, axis=1)
            results.append(cp.argsort(distances)[:K].get())
    
    return cp.array(results).T

def test_ann_methods():
    """Test and compare different ANN methods with different distance metrics"""
    print("\nTesting different ANN methods...")
    N, D, A, X, K = testdata_ann(args.testfile)
    
    # Get the distance metric from args
    metric = process_distance_func(args.dist) if hasattr(args, 'dist') else 'l2'
    print(f"\nTesting with {metric} distance:")
    
    # Get ground truth
    print("Computing exact KNN (ground truth)...")
    exact_results = our_knn(N, D, A, X, K, metric=metric)
    
    methods = [
        ("Basic ANN", our_ann_basic),
        ("K-means ANN", our_ann_kmeans),
        ("LSH ANN", our_ann_lsh)
    ]
    
    for name, method in methods:
        print(f"\nTesting {name}...")
        start_time = time.time()
        try:
            results = method(N, D, A, X, K, metric=metric)
            elapsed = time.time() - start_time
            
            recalls = []
            for i in range(X.shape[0]):
                recall = recall_rate(exact_results[:, i], results[:, i])
                recalls.append(recall)
            
            mean_recall = np.mean(recalls)
            std_recall = np.std(recalls)
            
            print(f"Time: {elapsed:.4f} seconds")
            print(f"Recall ({metric}): {mean_recall:.4f} ± {std_recall:.4f}")
        except Exception as e:
            print(f"Error testing {name}: {str(e)}")
            continue

if __name__ == "__main__":
    if not init_gpu():
        raise SystemExit("GPU initialization failed")
    
    print("\nTesting different ANN implementations...")
    test_ann_methods() 