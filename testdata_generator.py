import numpy as np
import argparse
import time

def recall_rate(exact_neighbors, approx_neighbors):
    """Calculate recall rate between exact and approximate neighbors"""
    return len(set(exact_neighbors) & set(approx_neighbors)) / len(exact_neighbors)

def generate_test_data(generate_big=False):
    # Different dimensions to test
    dimensions = [2, 64, 256, 1024]
    
    # Different dataset sizes
    sizes = [4000, 40000]  # Default sizes
    
    if generate_big:
        sizes.append(4000000)  # Add very large size only if flag is set
        print("\nWARNING: Generating very large files. This might use a lot of memory!")
    
    # Number of query points for X
    M = 10  # Generate 10 query points
    
    # Generate matrices
    for D in dimensions:
        # Generate X as a matrix (M x D) instead of vector
        X = np.random.randn(M, D)  # Matrix of M query points, each with D dimensions
        np.save(f'vector_D={D}.npy', X)
        print(f"Generated vector_D={D}.npy with shape ({M}, {D})")
        
        for N in sizes:
            # Skip very large high-dimensional combinations unless explicitly requested
            if N == 4000000 and D >= 256 and not generate_big:
                print(f"Skipping large matrix N={N}, D={D} (use --big to generate)")
                continue
                
            print(f"Generating matrix_D={D}_N={N}.npy")
            matrix = np.random.randn(N, D)
            np.save(f'matrix_D={D}_N={N}.npy', matrix)
            print(f"Done: matrix_D={D}_N={N}.npy with shape ({N}, {D})")

def test_recall(D=64, N=4000, K=10):
    """Test recall between KNN and ANN on generated data"""
    print(f"\nTesting recall for D={D}, N={N}, K={K}")
    
    # Load data
    try:
        A = np.load(f'matrix_D={D}_N={N}.npy')
        X = np.load(f'vector_D={D}.npy')
        print(f"Loaded A: {A.shape}, X: {X.shape}")
    except FileNotFoundError:
        print("Files not found. Generate test data first.")
        return
    
    # Run KNN (exact)
    start_time = time.time()
    distances = np.linalg.norm(A[:, None, :] - X[None, :, :], axis=2)
    exact_indices = np.argsort(distances, axis=0)[:K]
    knn_time = time.time() - start_time
    print(f"KNN time: {knn_time:.4f} seconds")
    
    # Run ANN (approximate)
    start_time = time.time()
    # Simple ANN implementation for testing
    num_clusters = min(int(np.sqrt(N)), 10)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(A)
    centroids = kmeans.cluster_centers_
    
    approx_indices = []
    for x in X:
        # Find nearest clusters
        centroid_distances = np.linalg.norm(centroids - x, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[:2]
        
        # Get points from nearest clusters
        candidates = []
        for c in nearest_clusters:
            cluster_points = A[cluster_labels == c]
            cluster_indices = np.where(cluster_labels == c)[0]
            distances = np.linalg.norm(cluster_points - x, axis=1)
            nearest = np.argsort(distances)[:K]
            candidates.extend(cluster_indices[nearest])
        
        # Get final K nearest from candidates
        candidate_distances = np.linalg.norm(A[candidates] - x, axis=1)
        final_indices = np.array(candidates)[np.argsort(candidate_distances)[:K]]
        approx_indices.append(final_indices)
    
    ann_time = time.time() - start_time
    print(f"ANN time: {ann_time:.4f} seconds")
    
    # Calculate recall
    recalls = []
    for i in range(len(X)):
        recall = recall_rate(exact_indices[:, i], approx_indices[i])
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    print(f"Average recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Speedup: {knn_time/ann_time:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--big', action='store_true', 
                      help='Generate very large files (4M points, might crash your machine)')
    parser.add_argument('--test', action='store_true',
                      help='Run recall test after generating data')
    args = parser.parse_args()
    
    # Create testdata directory if it doesn't exist
    import os
    os.makedirs('testdata', exist_ok=True)
    
    print("Generating test data...")
    print("Memory usage warning:")
    print("- Small  (4K points):  ~2MB per dimension")
    print("- Medium (40K points): ~80MB per dimension")
    if args.big:
        print("- Large  (4M points):  ~8GB per dimension")
    
    generate_test_data(args.big)
    print("\nDone generating data!")
    
    if args.test:
        # Test with different sizes
        test_recall(D=64, N=4000, K=10)    # Small
        test_recall(D=256, N=40000, K=10)  # Medium