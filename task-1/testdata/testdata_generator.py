import os
import numpy as np

# Parameters
N = [4000, 40000, 4000000]  # Number of data points. "Default" is 1000
D = [2, 64, 256, 1024]   # Dimensionality of the data. "Default is 100"

GENERATE_REALLY_BIG_FILES = False # Set to true if you want to generate the really big 4.000.000x256 and 4.000.000x1024 files (might crash your machine)

np.random.seed(42)

for n in N:
    for d in D:
        # File names
        matrix_file = f"matrix_D={d}_N={n}.npy"
        vector_file = f"vector_D={d}.npy"
        # Check if files already exist
        if os.path.exists(matrix_file):
            print(f"File '{matrix_file}' already exists. Skipping matrix generation.")
        elif(GENERATE_REALLY_BIG_FILES or (not (n==4000000 and d>64))):
            # Generate random data for A (N x D matrix)
            A = np.random.randn(n, d)
            # Save matrix A
            np.save(matrix_file, A)
            print(f"Matrix saved to '{matrix_file}'.")

        if os.path.exists(vector_file):
            print(f"File '{vector_file}' already exists. Skipping vector generation.")
        else:
            # Generate random query point X (D-dimensional vector)
            X = np.random.randn(d)
            # Save vector X
            np.save(vector_file, X)
            print(f"Vector saved to '{vector_file}'.")

print("Data file generation process completed.")