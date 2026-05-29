import numpy as np

from config import Nt, SAVE_VIDEO, VIDEO_PATH
from dynamics import bicycle_dynamics_numpy
from obstacles import obstacle_dynamics_all
from scenarios import scenario_dynamic_obstacles
from nmpc_controller import NMPCController
from plotting import plot_results
from mujoco_visualizer import save_mujoco_video, visualize_in_mujoco


def run_nmpc_simulation():
    controller = NMPCController()

    x, obs_states = scenario_dynamic_obstacles()

    x_hist = [x.copy()]
    obs_hist = [obs_states.copy()]
    u_hist = []

    for i in range(Nt):
        print(f"Solving NMPC step {i + 1}/{Nt}")

        u = controller.solve(x, obs_states)

        x = bicycle_dynamics_numpy(x, u)
        obs_states = obstacle_dynamics_all(obs_states)

        u_hist.append(u.copy())
        x_hist.append(x.copy())
        obs_hist.append(obs_states.copy())

    return np.array(x_hist), np.array(obs_hist), np.array(u_hist)


if __name__ == "__main__":
    x_hist, obs_hist, u_hist = run_nmpc_simulation()

    plot_results(x_hist, obs_hist, u_hist)

    if SAVE_VIDEO:
        save_mujoco_video(x_hist, obs_hist, u_hist, VIDEO_PATH)

    visualize_in_mujoco(x_hist, obs_hist, u_hist)