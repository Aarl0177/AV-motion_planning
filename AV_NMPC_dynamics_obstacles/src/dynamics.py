import numpy as np
import casadi as ca

from config import dt, L


def bicycle_dynamics_casadi(state, u):
    """
    CasADi symbolic kinematic bicycle model.
    state = [x, y, psi, v]
    u = [a, delta]
    """
    x, y, psi, v = state[0], state[1], state[2], state[3]
    a, delta = u[0], u[1]

    x_next = x + dt * v * ca.cos(psi)
    y_next = y + dt * v * ca.sin(psi)
    psi_next = psi + dt * v / L * ca.tan(delta)
    v_next = v + dt * a

    return ca.vertcat(x_next, y_next, psi_next, v_next)


def bicycle_dynamics_numpy(state, u):
    """
    Numeric kinematic bicycle model.
    """
    x, y, psi, v = state
    a, delta = u

    x_next = x + dt * v * np.cos(psi)
    y_next = y + dt * v * np.sin(psi)
    psi_next = psi + dt * v / L * np.tan(delta)
    v_next = v + dt * a

    return np.array([x_next, y_next, psi_next, v_next])