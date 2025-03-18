def our_ann(N, D, A, X, K, distance_func, k1=None, k2=None):
    """
    Approximate Nearest Neighbor search with k-means clustering
    
    Args:
        N: Number of database points
        D: Dimensions of points
        A: Database points (N, D)
        X: Query points (M, D)
        K: Number of nearest neighbors to find
        distance_func: Distance function to use
        k1: Number of clusters (if None, computed based on N)
        k2: Number of candidates per cluster (if None, computed based on K)
    """
    # Convert to GPU arrays
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    # Compute K1 (number of clusters) if not provided
    if k1 is None:
        k1 = min(int(np.sqrt(N) * 3), N//5)  # Default: sqrt(N) * 3
    K1 = max(5, min(k1, N//2))  # Ensure reasonable bounds
    
    # Compute K2 (candidates per cluster) if not provided
    if k2 is None:
        k2 = min(K * 5, N//5)  # Default: K * 5
    K2 = max(K * 2, min(k2, N//3))  # Ensure enough candidates
    
    # Run k-means clustering
    cluster_labels = our_kmeans(N, D, A_gpu, K1)
    centroids = cp.zeros((K1, D))
    
    # Compute centroids
    for k in range(K1):
        mask = cluster_labels == k
        if cp.any(mask):
            centroids[k] = cp.mean(A_gpu[mask], axis=0)
    
    # Process each query point
    results = []
    for x in X_gpu:
        # Find closest clusters
        centroid_distances = distance_func(centroids, x)
        closest_clusters = cp.argsort(centroid_distances)[:3]  # Check 3 closest clusters
        
        # Gather candidates from closest clusters
        candidates = []
        for cluster_idx in closest_clusters:
            cluster_points = A_gpu[cluster_labels == cluster_idx]
            cluster_indices = cp.where(cluster_labels == cluster_idx)[0]
            
            if len(cluster_points) == 0:
                continue
            
            # Find nearest points in cluster
            distances = distance_func(cluster_points, x)
            nearest = cp.argsort(distances)[:K2]
            candidates.extend(cluster_indices[nearest].get().tolist())
        
        # Final refinement
        if candidates:
            candidates = cp.array(candidates)
            final_distances = distance_func(A_gpu[candidates], x)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            # Fallback to full search
            distances = distance_func(A_gpu, x)
            indices = cp.argsort(distances)[:K]
            results.append(indices.get())
    
    return cp.array(results).T 