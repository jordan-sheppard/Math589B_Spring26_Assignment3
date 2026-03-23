# 
# In this module implement these two functions:
# 1. solve_continuous_are
# 2. solve_ivp
#
# Make sure that they are compatible with their usage
# in modal_lqr.py.
#
# The Gradescope Autograder will call your implementation
# through functions:
#
# 1. simulate_closed_loop
# 2. simulate_open_loop

import numpy as np

def solve_continous_are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the continuous Algebraic Ricatti Equation using the Direct method.

    Args:
        A, B : System matrices for the state dynamics
        Q, R : Weight matrices for the state and input in the LQR cost function

    Returns:
        P: Solution to the ARE
    """
    # Construct Hamilton matrix
    H = np.block(
        [
            [A , -B @ np.linalg.inv(R) @ B.T],
            [-Q, -A.T],
        ]
    )

    # Find stable eigenvectors of H
    eigenvals, eigenvecs = np.linalg.eig(A)
    stable_indices = np.real(eigenvals) < 0
    stable_eigenvecs = eigenvecs[:,stable_indices]

    # Partition
    n = A.shape[0]
    V_s1 = stable_eigenvecs[:n, :]
    V_s2 = stable_eigenvecs[n:, :]

    # Solve for P
    P = V_s2 @ np.linalg.inv(V_s1)
    return (P + P.T) / 2            # Enforce symmetry numerically