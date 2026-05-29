import numpy as np


def scenario_dynamic_obstacles():
    """
    Initial vehicle state and dynamic obstacles.
    """

    # Vehicle state: [x, y, psi, v]
    x0 = np.array([0.0, 0.0, 0.0, 2.0])

    # Obstacles: [x, y, vx, vy]
    obs_states = np.array([
        [3.0, 1.50,  0.30, 0.05],
        [10.0, 2.65, -0.25, 0.04],
        [20.0, 1.65, -0.50, 0.00],
    ])

    return x0, obs_states