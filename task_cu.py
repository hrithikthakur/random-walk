def our_ann(N, D, A, X, K, distance_func):
    """
    Improved ANN implementation based on PyTorch version
    """
    # Convert to GPU arrays
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    
    # Fixed hyperparameters
    K1 = 4  # Number of clusters to check
    K2 = 2000  # Candidates per cluster
    
    # Run k-means clustering
    cluster_labels = cp.asarray(our_kmeans(N, D, A_gpu, K))  # Ensure CuPy array
    centroids = cp.zeros((K, D))
    
    # Compute centroids
    for k in range(K):
        mask = cluster_labels == k
        mask = cp.asarray(mask)  # Convert mask to CuPy array
        if cp.any(mask):
            centroids[k] = cp.mean(A_gpu[mask], axis=0)
    
    # Process each query point
    results = []
    for x in X_gpu:
        # Find K1 closest clusters
        centroid_distances = distance_func(centroids, x)
        closest_clusters = cp.argsort(centroid_distances)[:K1]
        
        # Gather candidates from all selected clusters
        candidate_indices_list = []
        candidate_vectors_list = []
        
        for cluster_idx in closest_clusters:
            indices = cp.where(cluster_labels == cluster_idx)[0]
            points = A_gpu[indices]
            
            if len(points) > 0:
                distances = distance_func(points, x)
                nearest = cp.argsort(distances)[:min(len(points), K2)]
                candidate_indices_list.append(indices[nearest])
                candidate_vectors_list.append(points[nearest])
        
        # Process all candidates together
        if candidate_indices_list:
            candidates = cp.concatenate(candidate_indices_list)
            candidate_points = cp.concatenate(candidate_vectors_list)
            
            # Find final top-K
            final_distances = distance_func(candidate_points, x)
            final_indices = candidates[cp.argsort(final_distances)[:K]]
            results.append(final_indices.get())
        else:
            # Fallback to full search
            distances = distance_func(A_gpu, x)
            indices = cp.argsort(distances)[:K]
            results.append(indices.get())
    
    return cp.array(results).T