import numpy as np
import casadi as ca

from config import (
    N, dt, Q, R, Qf, x_dim, u_dim,
    a_min, a_max, steer_min, steer_max,
    safe_radius
)
from dynamics import bicycle_dynamics_casadi
from obstacles import predict_obstacle


class NMPCController:
    def __init__(self):
        pass

    def ref_path(self, state):
        """
        Reference path:
        vehicle should move forward along x while tracking y = 2.
        """
        y_ref = 2.0
        psi_ref = 0.0
        v_ref = 1.0

        X_ref = np.zeros((x_dim, N + 1))

        for k in range(N + 1):
            x_ref = state[0] + dt * v_ref * k
            X_ref[:, k] = np.array([x_ref, y_ref, psi_ref, v_ref])

        return X_ref

    def solve(self, X_0, obs_states):
        opti = ca.Opti()

        X = opti.variable(x_dim, N + 1)
        U = opti.variable(u_dim, N)

        num_obs = obs_states.shape[0]
        S = opti.variable(num_obs, N + 1)

        opti.subject_to(X[:, 0] == X_0)
        opti.subject_to(opti.bounded(0, S, ca.inf))

        X_ref = self.ref_path(X_0)
        obs_preds = [predict_obstacle(obs_states[j]) for j in range(num_obs)]

        cost = 0

        for k in range(N):
            u_k = U[:, k]
            X_next = bicycle_dynamics_casadi(X[:, k], u_k)
            opti.subject_to(X[:, k + 1] == X_next)

            for j in range(num_obs):
                obs_x_k = obs_preds[j][0, k]
                obs_y_k = obs_preds[j][1, k]

                dist_sq = (X[0, k] - obs_x_k) ** 2 + (X[1, k] - obs_y_k) ** 2

                opti.subject_to(dist_sq >= safe_radius**2 - S[j, k])
                cost += 1e6 * S[j, k] ** 2

            e = X[:, k] - X_ref[:, k]
            cost += e.T @ Q @ e
            cost += u_k.T @ R @ u_k

            opti.subject_to(opti.bounded(a_min, U[0, k], a_max))
            opti.subject_to(opti.bounded(steer_min, U[1, k], steer_max))

        e_N = X[:, N] - X_ref[:, N]
        cost += e_N.T @ Qf @ e_N

        for j in range(num_obs):
            obs_x_N = obs_preds[j][0, N]
            obs_y_N = obs_preds[j][1, N]

            dist_sq_N = (X[0, N] - obs_x_N) ** 2 + (X[1, N] - obs_y_N) ** 2

            opti.subject_to(dist_sq_N >= safe_radius**2 - S[j, N])
            cost += 1e6 * S[j, N] ** 2

        opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }

        opti.solver("ipopt", opts)

        opti.set_initial(X, X_ref)
        opti.set_initial(U, 0)
        opti.set_initial(S, 0.1)

        try:
            result = opti.solve()
            return result.value(U[:, 0])
        except RuntimeError:
            print("NMPC solver failed. Applying zero control.")
            print("IPOPT status:", opti.debug.return_status())
            return np.array([0.0, 0.0])