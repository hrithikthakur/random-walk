def distance_l2(X, Y):
    """
    Compute L2 distance between points in X and Y
    Ensures proper array dimensions
    """
    # Ensure Y is 2D if it's a single point
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    
    # Compute distances using broadcasting
    XX = cp.sum(X ** 2, axis=1, keepdims=True)
    YY = cp.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * cp.dot(X, Y.T)
    return cp.sqrt(distances)

def our_ann(N, D, A, X, K, distance_func):
    """
    Vectorized ANN implementation with batch distance calculations
    """
    # Convert to GPU arrays
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    M = X_gpu.shape[0]  # Number of query points
    
    # Fixed hyperparameters
    K1 = 4  # Number of clusters to check
    K2 = 2000  # Candidates per cluster
    
    # Run k-means clustering
    cluster_labels = cp.asarray(our_kmeans(N, D, A_gpu, K))
    centroids = cp.zeros((K, D))
    
    # Compute centroids
    for k in range(K):
        mask = cluster_labels == k
        if cp.any(mask):
            centroids[k] = cp.mean(A_gpu[mask], axis=0)
    
    # Calculate all centroid distances at once for all query points
    centroid_distances = distance_func(centroids, X_gpu)  # Shape: (K, M)
    closest_clusters = cp.argsort(centroid_distances, axis=0)[:K1]  # Shape: (K1, M)
    
    # Process all query points at once
    results = []
    for i in range(M):
        clusters = closest_clusters[:, i]
        
        # Gather candidates from selected clusters
        candidate_indices = []
        for cluster_idx in clusters:
            indices = cp.where(cluster_labels == cluster_idx)[0]
            if len(indices) > 0:
                points = A_gpu[indices]
                # Calculate distances to all points in cluster at once
                distances = distance_func(points, X_gpu[i:i+1])  # Keep 2D shape
                nearest = cp.argsort(distances.ravel())[:min(len(indices), K2)]
                candidate_indices.append(indices[nearest])
        
        if candidate_indices:
            # Concatenate all candidates
            candidates = cp.concatenate(candidate_indices)
            candidate_points = A_gpu[candidates]
            
            # Calculate final distances all at once
            final_distances = distance_func(candidate_points, X_gpu[i:i+1]).ravel()
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            # Fallback: calculate all distances at once
            distances = distance_func(A_gpu, X_gpu[i:i+1]).ravel()
            indices = cp.argsort(distances)[:K]
            results.append(indices.get())
    
    return cp.array(results).T

def test_ann_detailed():
    # ... existing code ...
    
    # Fix the match rate calculation
    match_rate = np.mean([
        len(set(r1.tolist()) & set(r2.tolist())) / K 
        for r1, r2 in zip(exact_results.T, ann_results.T)
    ])
    
    # ... rest of the code ...