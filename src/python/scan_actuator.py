"""Compare actuator locations for the truncated membrane model."""

from __future__ import annotations

import numpy as np

from .modal_lqr import build_model


def summarize_location(x0: float, y0: float, M: int = 6) -> None:
    model = build_model(M=M, x0=x0, y0=y0)
    small = np.sum(np.abs(model.beta) < 1e-8)
    print(f"location=({x0:.3f},{y0:.3f})  min|beta|={np.min(np.abs(model.beta)):.3e}  zero-like modes={small}")


def main() -> None:
    test_locations = [
        (0.50, 0.50),
        (0.37, 0.61),
        (0.25, 0.50),
        (0.21, 0.29),
    ]
    for x0, y0 in test_locations:
        summarize_location(x0, y0)

    print("\nTiny couplings can also be found by scanning a dense grid of actuator locations.")
    grid = np.linspace(0.1, 0.9, 9)
    best = None
    for x0 in grid:
        for y0 in grid:
            model = build_model(M=5, x0=float(x0), y0=float(y0))
            score = float(np.min(np.abs(model.beta)))
            if best is None or score > best[0]:
                best = (score, x0, y0)
    assert best is not None
    score, x0, y0 = best
    print(f"Best coarse-grid location by maximin coupling: ({x0:.3f}, {y0:.3f}), score={score:.3e}")


if __name__ == "__main__":
    main()
