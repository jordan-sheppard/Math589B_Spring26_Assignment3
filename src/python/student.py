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



def solve_continuous_are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
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
    eigenvals, eigenvecs = np.linalg.eig(H)
    stable_indices = np.real(eigenvals) < 0
    stable_eigenvecs = eigenvecs[:,stable_indices]

    # Partition
    n = A.shape[0]
    V_s1 = stable_eigenvecs[:n, :]
    V_s2 = stable_eigenvecs[n:, :]

    # Solve for P
    P = V_s2 @ np.linalg.inv(V_s1)
    return np.real((P + P.T) / 2)            # Enforce symmetry numerically


def solve_ivp(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: Optional[np.ndarray] =None,
    rtol: Optional[float] = 1e-8, 
    atol: Optional[float] = 1e-10
) -> ODESolution:
    """
    Solves the given IVP using the DP5 integration method.

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
    STEP_SAFETY_FACTOR = 0.6           # Factor to limit rejection for steps being too large

    t = t_span[0]
    y = y0.copy()
    t_vals = [t]
    y_vals = [y.copy()]
    h = t_eval[1] - t_eval[0] if t_eval is not None else 1e-4
    while t < t_span[1]:        
        # Dormand-Prince 5(4) integration scheme
        k1 = h * f(t, y)
        k2 = h * f(t + (1/5)*h, y + (1/5)*k1)
        k3 = h * f(t + (3/10)*h, y + (3/40)*k1 + (9/40)*k2)
        k4 = h * f(t + (4/5)*h, y + (44/45)*k1 - (56/15)*k2 + (32/9)*k3)
        k5 = h * f(t + (8/9)*h, y + (19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4)
        k6 = h * f(t + h, y + (9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5)
        k7 = h * f(t + h, y + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)
        
        # The 5th-order update (which DP5 optimizes for)
        y5 = y + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6
        
        # The 4th-order update (used only for error estimation)
        y4 = y + (5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 - (92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7
        
        # Compute error 
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        error_ratio = (y5 - y4) / scale
        err = np.sqrt(np.mean(error_ratio**2))
        if err <= 1.0:
            # ACCEPT STEP SIZE
            # Update for next iteration
            t += h
            y = y5.copy()
            h = h * min(2, (1/err)**(0.2)) * STEP_SAFETY_FACTOR

            # Record state and time of this iteration
            t_vals.append(t)
            y_vals.append(y)
        else:
            # REJECT STEP SIZE
            h = h * max(0.5, (1/err)**(0.2)) * STEP_SAFETY_FACTOR

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




