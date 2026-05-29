import numpy as np

from config import dt, N


def predict_obstacle(obs_state):
    """
    Constant-velocity obstacle prediction.
    obs_state = [ox, oy, ovx, ovy]
    """
    ox, oy, ovx, ovy = obs_state
    obs_pred = np.zeros((2, N + 1))

    for k in range(N + 1):
        obs_pred[0, k] = ox + ovx * dt * k
        obs_pred[1, k] = oy + ovy * dt * k

    return obs_pred


def obstacle_dynamics_all(obs_states):
    """
    Update all obstacles with constant velocity.
    obs_states[j] = [ox, oy, ovx, ovy]
    """
    obs_next = obs_states.copy()
    obs_next[:, 0] = obs_states[:, 0] + dt * obs_states[:, 2]
    obs_next[:, 1] = obs_states[:, 1] + dt * obs_states[:, 3]

    return obs_next