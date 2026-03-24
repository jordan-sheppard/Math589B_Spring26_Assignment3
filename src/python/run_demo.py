"""Generate plots and optional animation frames for the square membrane LQR project."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from .modal_lqr import (
    build_lqr,
    compute_energy,
    demo_configuration,
    ensure_dir,
    reconstruct_field,
    summarize_couplings,
    simulate_closed_loop,
    simulate_open_loop,
)


def save_energy_plot(outdir: Path, t_open, e_open, t_closed, e_closed) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_open, e_open, label="open loop")
    plt.plot(t_closed, e_closed, label="closed loop")
    plt.xlabel("time")
    plt.ylabel("modal energy")
    plt.title("Energy decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "energy.png", dpi=160)
    plt.close()


def save_control_plot(outdir: Path, t, u) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(t, u)
    plt.xlabel("time")
    plt.ylabel("control input")
    plt.title("LQR control signal")
    plt.tight_layout()
    plt.savefig(outdir / "control.png", dpi=160)
    plt.close()


def save_snapshots(outdir: Path, model, t, y, times=(0.0, 0.5, 1.5, 3.0, 6.0)) -> None:
    N = len(model.modes)
    for target in times:
        j = int(np.argmin(np.abs(t - target)))
        X, Y, U = reconstruct_field(model, y[:N, j], grid_size=81)
        plt.figure(figsize=(5.3, 4.5))
        plt.contourf(X, Y, U, levels=30)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Membrane displacement at t={t[j]:.2f}")
        plt.tight_layout()
        plt.savefig(outdir / f"snapshot_t_{t[j]:.2f}.png", dpi=160)
        plt.close()


def save_animation(outdir: Path, model, t, y, nframes: int = 80) -> None:
    N = len(model.modes)
    frame_ids = np.linspace(0, len(t) - 1, nframes).astype(int)
    _, _, U0 = reconstruct_field(model, y[:N, frame_ids[0]], grid_size=81)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(U0.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", animated=True)
    fig.colorbar(im)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"t={t[frame_ids[0]]:.2f}")

    vmax = max(1e-8, np.max(np.abs(y[:N, :]))) * 4.0
    im.set_clim(-vmax, vmax)

    def update(frame_index: int):
        _, _, U = reconstruct_field(model, y[:N, frame_ids[frame_index]], grid_size=81)
        im.set_array(U.T)
        title.set_text(f"t={t[frame_ids[frame_index]]:.2f}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=80, blit=True)
    try:
        ani.save(outdir / "membrane.gif", writer="pillow", fps=12)
    except Exception as exc:  # pragma: no cover
        print(f"Could not save GIF animation: {exc}")
    plt.close(fig)

def main() -> None:
    outdir = ensure_dir(Path(__file__).resolve().parents[2] / "outputs")
    model, x_init = demo_configuration()
    print(summarize_couplings(model))

    _, _, _, K = build_lqr(model, alpha=1.0, beta_v=1.0, R=5e-2)
    t_open, y_open = simulate_open_loop(model, x_init, T=6.0, nt=500)
    t_closed, y_closed, u_closed = simulate_closed_loop(model, K, x_init, T=6.0, nt=500)

    e_open = compute_energy(model, y_open)
    e_closed = compute_energy(model, y_closed)

    print()
    print(f"Initial energy:       {e_closed[0]:.6e}")
    print(f"Final open energy:    {e_open[-1]:.6e}")
    print(f"Final closed energy:  {e_closed[-1]:.6e}")
    print(f"Max |control|:        {np.max(np.abs(u_closed)):.6e}")

    # save_energy_plot(outdir, t_open, e_open, t_closed, e_closed)
    # save_control_plot(outdir, t_closed, u_closed)
    # save_snapshots(outdir, model, t_closed, y_closed)
    # save_animation(outdir, model, t_closed, y_closed)
    # print(f"Wrote demo artifacts to {outdir}")


if __name__ == "__main__":
    main()
