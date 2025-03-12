import numpy as np
import argparse
import os

def generate_test_data(generate_big=False):
    dimensions = [2, 64, 256, 1024]
    sizes = [4000, 40000]
    
    if generate_big:
        sizes.append(4000000)
        print("\nWARNING: Generating very large files. This might use a lot of memory!")
    
    M = 10  # number of query points
    
    for D in dimensions:
        X = np.random.randn(M, D)
        np.save(f'vector_D={D}.npy', X)
        print(f"Generated vector_D={D}.npy with shape ({M}, {D})")
        
        for N in sizes:
            if N == 4000000 and D >= 256 and not generate_big:
                print(f"Skipping large matrix N={N}, D={D} (use --big to generate)")
                continue
            
            print(f"Generating matrix_D={D}_N={N}.npy")
            matrix = np.random.randn(N, D)
            np.save(f'matrix_D={D}_N={N}.npy', matrix)
            print(f"Done: matrix_D={D}_N={N}.npy with shape ({N}, {D})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--big', action='store_true', help='Generate very large files (4M points, might crash your machine)')
    args = parser.parse_args()
    
    os.makedirs('testdata', exist_ok=True)
    
    print("Generating test data...")
    print("Memory usage warning:")
    print("- Small  (4K points):  ~2MB per dimension")
    print("- Medium (40K points): ~80MB per dimension")
    if args.big:
        print("- Large  (4M points):  ~8GB per dimension")
    
    generate_test_data(args.big)
    print("\nDone!")