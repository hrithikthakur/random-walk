import numpy as np
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--big', action='store_true', 
                      help='Generate very large files (4M points, might crash your machine)')
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
    print("\nDone!")