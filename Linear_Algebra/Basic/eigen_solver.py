import numpy as np

def compute_eigen(matrix_str):
    """
    Computes and returns the eigenvalues and eigenvectors for a given square matrix string.
    
    Parameters:
        matrix_str (str): A string representation of a square matrix (e.g., '[[4, 2], [1, 3]]')
        
    Returns:
        tuple: (eigenvalues, eigenvectors) or error message if input is invalid
    """
    try:
        A = np.array(eval(matrix_str), dtype=float)
        
        if A.shape[0] != A.shape[1]:
            return "Error: Matrix must be square."
        
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors

    except Exception as e:
        return f"Invalid input or error: {e}"


# Optional main block for standalone use
if __name__ == "__main__":
    # matrix_input = input("Enter a square matrix (e.g., [[4, 2], [1, 3]]): ")
    matrix_input = '[[4, 2], [1, 3]]'  # Example input for testing
    result = compute_eigen(matrix_input)

    if isinstance(result, str):
        print(result)
    else:
        eigenvalues, eigenvectors = result
        print("\nEigenvalues:")
        print(eigenvalues)
        print("\nEigenvectors (column-wise):")
        print(eigenvectors)
