import numpy as np
import matplotlib.pyplot as plt

from config import dt, obs_radius


def plot_results(x_hist, obs_hist, u_hist):
    """
    Plot controls, states, and x-y trajectories.
    """

    t_state = np.arange(len(x_hist)) * dt
    t_control = np.arange(len(u_hist)) * dt

    x = x_hist[:, 0]
    y = x_hist[:, 1]
    psi = x_hist[:, 2]
    v = x_hist[:, 3]

    a = u_hist[:, 0]
    delta = u_hist[:, 1]

    # Controls
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_control, a, linewidth=2)
    plt.ylabel("Acceleration a [m/s²]")
    plt.grid(True)
    plt.title("NMPC Control Inputs")

    plt.subplot(2, 1, 2)
    plt.plot(t_control, delta, linewidth=2)
    plt.ylabel("Steering angle δ [rad]")
    plt.xlabel("Time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("controls_plot.png", dpi=300)
    plt.show()

    # States
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_state, v, linewidth=2)
    plt.ylabel("Velocity v [m/s]")
    plt.grid(True)
    plt.title("Vehicle States")

    plt.subplot(2, 1, 2)
    plt.plot(t_state, psi, linewidth=2)
    plt.ylabel("Yaw angle ψ [rad]")
    plt.xlabel("Time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("vehicle_states_plot.png", dpi=300)
    plt.show()

    # X-Y trajectory
    plt.figure(figsize=(11, 6))

    plt.plot(x, y, linewidth=3, label="Vehicle trajectory")

    plt.scatter(x[0], y[0], s=80, marker="o", label="Start")
    plt.scatter(x[-1], y[-1], s=80, marker="x", label="End")

    plt.axhline(y=2.0, linestyle="--", linewidth=2, label="Reference path y = 2")

    num_obs = obs_hist.shape[1]

    for j in range(num_obs):
        ox = obs_hist[:, j, 0]
        oy = obs_hist[:, j, 1]

        plt.plot(ox, oy, linewidth=2, linestyle="--", label=f"Obstacle {j + 1} path")
        plt.scatter(ox[0], oy[0], s=60, marker="o")
        plt.scatter(ox[-1], oy[-1], s=60, marker="x")

        circle = plt.Circle(
            (ox[-1], oy[-1]),
            obs_radius,
            fill=False,
            linewidth=2
        )
        plt.gca().add_patch(circle)

    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("Vehicle x-y Trajectory and Obstacle Paths")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("xy_trajectory_plot.png", dpi=300)
    plt.show()