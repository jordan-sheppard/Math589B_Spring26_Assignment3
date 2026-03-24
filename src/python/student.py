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
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

@dataclass
class ODESolution:
    """
    A numerical solution to a system of ODE's.
    
    Attributes:
        t (np.ndarray): (n_points,); The timestamp of each state
        y (np.ndarray): (n, n_points); The state at each timestep
            (y[:, t] is the state at time t)
    """
    t: np.ndarray
    y: np.ndarray



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


def solve_ivp(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: Optional[np.ndarray] =None,
    rtol: Optional[float] = 1e-8, 
    atol: Optional[float] = 1e-10
) -> ODESolution:
    """
    Solves the given IVP using a Runge-Kutta 45 integration method.

    Args:
        fun: A function with signature f(t, y); given a state and time, outputs
            the time derivative of the state:
            dy/dt = f(t, y)
        t_span: The time interval on which to solve the ODE
        y0: The initial state
        t_eval: The times at which to evaluate the final solution (uses a quartic
            interpolation polynomial to evaluate between adaptive timesteps
            of the integration scheme)
        rtol: The relative tolerance 
        atol: The absolute tolerance

    Returns:
        ODESolution: The state and timesteps of the numerical solution.
    """
    STEP_SAFETY_FACTOR = 0.9            # Factor to limit rejection for steps being too large

    t = t_span[0]
    y = y0.copy()
    t_vals = [t]
    y_vals = [y.copy()]
    h = t_eval[1] - t_eval[0] if t_eval is not None else 1e-4
    while t < t_span[1]:
        # RK45 integration scheme
        k1 = h * f(t, y)
        k2 = h * f(t + h/4, y + k1/4)
        k3 = h * f(t + (3/8 * h), y + (3/32 * k1) + (9/32 * k2))
        k4 = h * f(t + (12/13 * h), y + (1932/2197 * k1) - (7200/2197 * k2) + (7296/2197 * k3))
        k5 = h * f(t + h, y + (439/216 * k1) - (8 * k2) + (3680/513 * k3) - (845/4184 * k4))
        k6 = h * f(t + h/2, y - (8/27 * k1) + (2 * k2) - (3544/2565 * k3) + (1859/4104 * k4) - (11/40 * k5))
        y4 = y + (25/216 * k1) + (1408/2565 * k3) + (2197/4104 * k4) - (k5 / 5)
        y5 = y + (16/135 * k1 ) + (6656/12825 * k3) + (28561/56430 * k4) - (9/50 * k5) + (2/55 * k6)
        
        # Compute error 
        abserr = np.linalg.norm(y5 - y4)
        threshold = atol + rtol * np.linalg.norm(y5)    # Absolute error tolerance, plus relative error (with no division by norm)
        if abserr < threshold:
            # ACCEPT STEP SIZE
            # Update for next iteration
            t += h
            y = y5.copy()
            h = h * min(2, (threshold/abserr)**(0.2)) * STEP_SAFETY_FACTOR

            # Record state and time of this iteration
            t_vals.append(t)
            y_vals.append(y)
        else:
            # REJECT STEP SIZE
            h = h * max(0.5, (threshold/abserr)**(0.2)) * STEP_SAFETY_FACTOR

        # Truncate to t_end if we've gone too far 
        if t + h > t_span[1]:
            h = t_span[1] - t
    
    # If no provided times for interpolation, use RK45 timesteps
    if t_eval is None:
        t_eval = np.array(t_vals)
        y_eval = np.array(y_vals).T 
        return ODESolution(
            t=t_eval,
            y=y_eval
        )
    
    # Interpolate states at given evaluation times in t_eval
    y_eval = np.zeros((len(y0), len(t_eval)))
    time_idx = 0
    for i in range(1, len(t_vals)):
        # Cubic spline interpolant between these two points
        t1, t2 = t_vals[i-1], t_vals[i]
        y1, y2 = y_vals[i-1], y_vals[i]
        dy1, dy2 = f(t1, y1), f(t2, y2)
        while time_idx < (len(t_eval) - 1) and t_eval[time_idx] >= t1 and t_eval[time_idx] < t2:
            t = t_eval[time_idx]
            s = (t-t1)/(t2-t1)
            w1 = 2*(s**3) - 3*(s**2) + 1
            w2 = -2*(s**3) + 3*(s**2)
            w3 = (s**3) - 2*(s**2) + s
            w4 = (s**3) - (s**2)
            y_eval[:, time_idx] = (w1 * y1) + (w2 * y2) + (w3 * (t2 - t1) * dy1) + (w4 * (t2 - t1) * dy2)
            time_idx += 1

    y_eval[:, -1] = y_vals[-1] # Catch final state

    return ODESolution(
        t=t_eval,
        y=y_eval
    )




