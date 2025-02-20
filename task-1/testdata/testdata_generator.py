import os
import numpy as np

# Parameters
N = 1000  # Number of data points. "Default" is 1000
D = 100   # Dimensionality of the data. "Default is 100"

# File names
matrix_file = f"matrix_D={D}_N={N}.txt"
vector_file = f"vector_D={D}.txt"

# Check if files already exist
if os.path.exists(matrix_file):
    print(f"File '{matrix_file}' already exists. Skipping matrix generation.")
else:
    # Generate random data for A (N x D matrix)
    A = np.random.randn(N, D)
    # Save matrix A
    np.savetxt(matrix_file, A)
    print(f"Matrix saved to '{matrix_file}'.")

if os.path.exists(vector_file):
    print(f"File '{vector_file}' already exists. Skipping vector generation.")
else:
    # Generate random query point X (D-dimensional vector)
    X = np.random.randn(D)
    # Save vector X
    np.savetxt(vector_file, X)
    print(f"Vector saved to '{vector_file}'.")

print("Data file generation process completed.")