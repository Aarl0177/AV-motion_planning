

import time
import numpy as np
import casadi as ca
import mujoco
import mujoco.viewer
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# -----------------------------
# NMPC settings
# -----------------------------
N = 60
Nt = 160
T = 16.0
dt = T / Nt

# Bicycle model wheelbase
L = 1.0

a_min, a_max = -2.0, 2.0
steer_min, steer_max = -np.pi / 10, np.pi / 10

Q = np.diag([3, 30, 20, 2])
R = np.diag([5, 1])
Qf = np.diag([5, 5, 1, 1])

x_dim = Q.shape[0]
u_dim = R.shape[0]

# Obstacle / safety settings
obs_radius = 0.30
vehicle_radius = 0.20
margin = 0.40
safe_radius = obs_radius + vehicle_radius + margin


# -----------------------------
# Video saving settings
# -----------------------------
SAVE_VIDEO = True
VIDEO_PATH = "stage2_mujoco_car_lidar_simulation.mp4"
VIDEO_FPS = 10
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


class AV:
    def __init__(self, dt):
        self.dt = dt

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
            x_ref = state[0] + self.dt * v_ref * k
            X_ref[:, k] = np.array([x_ref, y_ref, psi_ref, v_ref])

        return X_ref

    def predict_obstacle(self, obs_state):
        """
        Constant-velocity obstacle prediction.
        obs_state = [ox, oy, ovx, ovy]
        """
        ox, oy, ovx, ovy = obs_state
        obs_pred = np.zeros((2, N + 1))

        for k in range(N + 1):
            obs_pred[0, k] = ox + ovx * self.dt * k
            obs_pred[1, k] = oy + ovy * self.dt * k

        return obs_pred

    def predict_dyn(self, state, u):
        """
        CasADi symbolic kinematic bicycle model.
        state = [x, y, psi, v]
        u = [a, delta]
        """
        x, y, psi, v = state[0], state[1], state[2], state[3]
        a, delta = u[0], u[1]

        x_next = x + self.dt * v * ca.cos(psi)
        y_next = y + self.dt * v * ca.sin(psi)
        psi_next = psi + self.dt * v / L * ca.tan(delta)
        v_next = v + self.dt * a

        return ca.vertcat(x_next, y_next, psi_next, v_next)

    def NMPC_solver(self, X_0, obs_states):
        opti = ca.Opti()

        X = opti.variable(x_dim, N + 1)
        U = opti.variable(u_dim, N)

        num_obs = obs_states.shape[0]
        S = opti.variable(num_obs, N + 1)

        opti.subject_to(X[:, 0] == X_0)
        opti.subject_to(opti.bounded(0, S, ca.inf))

        X_ref = self.ref_path(X_0)
        obs_preds = [self.predict_obstacle(obs_states[j]) for j in range(num_obs)]

        cost = 0

        for k in range(N):
            u_k = U[:, k]
            X_next = self.predict_dyn(X[:, k], u_k)
            opti.subject_to(X[:, k + 1] == X_next)

            # Obstacle avoidance with slack variables
            for j in range(num_obs):
                obs_x_k = obs_preds[j][0, k]
                obs_y_k = obs_preds[j][1, k]

                dist_sq = (X[0, k] - obs_x_k) ** 2 + (X[1, k] - obs_y_k) ** 2

                opti.subject_to(dist_sq >= safe_radius**2 - S[j, k])
                cost += 1e6 * S[j, k] ** 2

            # Tracking and control effort cost
            e = X[:, k] - X_ref[:, k]
            cost += e.T @ Q @ e
            cost += u_k.T @ R @ u_k

            opti.subject_to(opti.bounded(a_min, U[0, k], a_max))
            opti.subject_to(opti.bounded(steer_min, U[1, k], steer_max))

        # Terminal cost
        e_N = X[:, N] - X_ref[:, N]
        cost += e_N.T @ Qf @ e_N

        # Terminal obstacle constraints
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

    def dynamics(self, state, u):
        """
        Numeric kinematic bicycle model.
        """
        x, y, psi, v = state
        a, delta = u

        x_next = x + self.dt * v * np.cos(psi)
        y_next = y + self.dt * v * np.sin(psi)
        psi_next = psi + self.dt * v / L * np.tan(delta)
        v_next = v + self.dt * a

        return np.array([x_next, y_next, psi_next, v_next])


def obstacle_dynamics_all(obs_states, dt):
    """
    Update all obstacles with constant velocity.
    obs_states[j] = [ox, oy, ovx, ovy]
    """
    obs_next = obs_states.copy()
    obs_next[:, 0] = obs_states[:, 0] + dt * obs_states[:, 2]
    obs_next[:, 1] = obs_states[:, 1] + dt * obs_states[:, 3]

    return obs_next


# -----------------------------
# Quaternion helpers
# -----------------------------
def yaw_to_quat_z(yaw):
    """
    Rotation around z-axis.
    MuJoCo quaternion order: [w, x, y, z]
    """
    return np.array([
        np.cos(yaw / 2),
        0.0,
        0.0,
        np.sin(yaw / 2)
    ])


def quat_x(angle):
    """
    Rotation around x-axis.
    MuJoCo quaternion order: [w, x, y, z]
    """
    return np.array([
        np.cos(angle / 2),
        np.sin(angle / 2),
        0.0,
        0.0
    ])


def quat_mul(q1, q2):
    """
    Quaternion multiplication.
    MuJoCo quaternion order: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def build_mujoco_xml(num_obs):
    """
    Build MuJoCo XML scene.

    The car body is still kinematic/visual:
    we directly update its position and yaw from the bicycle model.
    """

    obstacle_bodies = "\n".join(
        f"""
        <body name="obs_{j}" pos="0 0 0.30">
            <geom name="obs_geom_{j}"
                  type="sphere"
                  size="{obs_radius}"
                  rgba="0.9 0.1 0.1 1"/>
        </body>

        <body name="safe_{j}" pos="0 0 0.015">
            <geom name="safe_geom_{j}"
                  type="cylinder"
                  size="{safe_radius} 0.015"
                  rgba="1 0 0 0.18"/>
        </body>
        """
        for j in range(num_obs)
    )

    return f"""
<mujoco model="nmpc_bicycle_car_lidar_visualization">

    <option timestep="{dt}" gravity="0 0 -9.81"/>

    <visual>
        <quality shadowsize="2048"/>
        <global offwidth="{VIDEO_WIDTH}" offheight="{VIDEO_HEIGHT}"/>
        <map znear="0.01" zfar="100"/>
    </visual>

    <asset>
        <texture name="grid"
                 type="2d"
                 builtin="checker"
                 width="512"
                 height="512"
                 rgb1="0.2 0.2 0.2"
                 rgb2="0.3 0.3 0.3"/>

        <material name="grid_mat"
                  texture="grid"
                  texrepeat="20 20"
                  reflectance="0.1"/>

        <material name="lane_green" rgba="0.1 0.8 0.1 1"/>
        <material name="car_blue" rgba="0.1 0.25 0.9 1"/>
        <material name="car_dark" rgba="0.03 0.05 0.12 1"/>
        <material name="wheel_black" rgba="0.01 0.01 0.01 1"/>
        <material name="lidar_blue" rgba="0.1 0.8 1.0 0.35"/>
    </asset>

    <worldbody>

        <light pos="0 0 8"
               dir="0 0 -1"
               diffuse="1 1 1"/>

        <camera name="video_cam"
                pos="9 -10 6"
                xyaxes="1 0 0 0 0.55 0.83"/>

        <geom name="ground"
              type="plane"
              size="40 8 0.1"
              material="grid_mat"/>

        <!-- Reference lane y = 2 -->
        <body name="reference_line" pos="10 2 0.025">
            <geom type="box"
                  size="20 0.025 0.01"
                  material="lane_green"/>
        </body>

        <!-- Visual car body. The bicycle model controls this body. -->
        <body name="vehicle" pos="0 0 0.22">

        <!-- Main chassis -->
        <geom name="vehicle_body"
            type="box"
            pos="0 0 0"
            size="0.35 0.18 0.08"
            material="car_blue"/>

        <!-- Hood/front -->
        <geom name="vehicle_hood"
            type="box"
            pos="0.22 0 0.03"
            size="0.12 0.15 0.05"
            rgba="0.2 0.48 1.0 1"/>

        <!-- Cabin -->
        <geom name="vehicle_cabin"
            type="box"
            pos="-0.06 0 0.11"
            size="0.14 0.13 0.07"
            material="car_dark"/>

        <!-- Small front sensor block -->
        <geom name="lidar_sensor"
            type="box"
            pos="0.38 0 0.08"
            size="0.025 0.05 0.025"
            rgba="0.0 0.0 0.0 1"/>

        <!-- Two LiDAR beams -->
        <geom name="lidar_left_beam"
            type="capsule"
            fromto="0.39 0.02 0.08 3.20 0.80 0.08"
            size="0.012"
            rgba="0.1 0.8 1.0 0.35"/>

        <geom name="lidar_right_beam"
            type="capsule"
            fromto="0.39 -0.02 0.08 3.20 -0.80 0.08"
            size="0.012"
            rgba="0.1 0.8 1.0 0.35"/>

        <!-- Front-left steering wheel -->
        <body name="front_left_wheel" pos="0.23 0.21 -0.07">
            <geom name="front_left_wheel_geom"
                type="cylinder"
                size="0.055 0.025"
                material="wheel_black"/>
        </body>

        <!-- Front-right steering wheel -->
        <body name="front_right_wheel" pos="0.23 -0.21 -0.07">
            <geom name="front_right_wheel_geom"
                type="cylinder"
                size="0.055 0.025"
                material="wheel_black"/>
        </body>

        <!-- Rear-left wheel -->
        <body name="rear_left_wheel" pos="-0.23 0.21 -0.07">
            <geom name="rear_left_wheel_geom"
                type="cylinder"
                size="0.055 0.025"
                material="wheel_black"/>
        </body>

        <!-- Rear-right wheel -->
        <body name="rear_right_wheel" pos="-0.23 -0.21 -0.07">
            <geom name="rear_right_wheel_geom"
                type="cylinder"
                size="0.055 0.025"
                material="wheel_black"/>
        </body>

    </body>

        {obstacle_bodies}

    </worldbody>

</mujoco>
"""


def run_nmpc_simulation():
    av = AV(dt)

    # Initial vehicle state: [x, y, psi, v]
    x = np.array([0.0, 0.0, 0.0, 2.0])

    # Obstacles: [x, y, vx, vy]
    obs_states = np.array([
        [3.0, 1.50,  0.30, 0.05],
        [10.0, 2.65, -0.25, 0.04],
        [20.0, 1.65, -0.50, 0.00],
    ])

    x_hist = [x.copy()]
    obs_hist = [obs_states.copy()]
    u_hist = []

    for i in range(Nt):
        print(f"Solving NMPC step {i + 1}/{Nt}")

        u = av.NMPC_solver(x, obs_states)

        x = av.dynamics(x, u)
        obs_states = obstacle_dynamics_all(obs_states, dt)

        u_hist.append(u.copy())
        x_hist.append(x.copy())
        obs_hist.append(obs_states.copy())

    return np.array(x_hist), np.array(obs_hist), np.array(u_hist)


def get_body_ids(model, num_obs):
    """
    Cache MuJoCo body IDs.
    """
    ids = {
        "vehicle": model.body("vehicle").id,
        "front_left_wheel": model.body("front_left_wheel").id,
        "front_right_wheel": model.body("front_right_wheel").id,
        "rear_left_wheel": model.body("rear_left_wheel").id,
        "rear_right_wheel": model.body("rear_right_wheel").id,
        "obs": [model.body(f"obs_{j}").id for j in range(num_obs)],
        "safe": [model.body(f"safe_{j}").id for j in range(num_obs)],
    }

    return ids


def set_replay_frame(model, data, ids, x_hist, obs_hist, u_hist, frame):
    """
    Update MuJoCo visual scene for one replay frame.
    """
    num_obs = obs_hist.shape[1]

    # Vehicle state
    x_pos, y_pos, psi, _ = x_hist[frame]

    # Steering input from NMPC
    if len(u_hist) == 0:
        delta = 0.0
    elif frame < len(u_hist):
        delta = u_hist[frame][1]
    else:
        delta = u_hist[-1][1]

    # Move vehicle according to bicycle model state
    model.body_pos[ids["vehicle"]] = np.array([x_pos, y_pos, 0.30])
    model.body_quat[ids["vehicle"]] = yaw_to_quat_z(psi)

    # Wheel orientation
    # Cylinder axis must be rotated to align like a car wheel.
    wheel_base_quat = quat_x(np.pi / 2)

    # Front wheels steer by delta
    front_wheel_quat = quat_mul(yaw_to_quat_z(delta), wheel_base_quat)

    model.body_quat[ids["front_left_wheel"]] = front_wheel_quat
    model.body_quat[ids["front_right_wheel"]] = front_wheel_quat

    # Rear wheels stay straight
    model.body_quat[ids["rear_left_wheel"]] = wheel_base_quat
    model.body_quat[ids["rear_right_wheel"]] = wheel_base_quat

    # Move obstacles and safety zones
    for j in range(num_obs):
        ox = obs_hist[frame, j, 0]
        oy = obs_hist[frame, j, 1]

        model.body_pos[ids["obs"][j]] = np.array([ox, oy, 0.30])
        model.body_pos[ids["safe"][j]] = np.array([ox, oy, 0.015])

    mujoco.mj_forward(model, data)


def save_mujoco_video(x_hist, obs_hist, u_hist, output_path=VIDEO_PATH):
    """
    Save MuJoCo replay as MP4 using the same camera angle style
    as the live viewer.
    """
    num_obs = obs_hist.shape[1]

    xml = build_mujoco_xml(num_obs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ids = get_body_ids(model, num_obs)

    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    writer = imageio.get_writer(output_path, fps=VIDEO_FPS, macro_block_size=1)

    # Create a free camera for saved video
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Match your live-view camera settings
    cam.distance = 13.5
    cam.azimuth = 10      # change this angle
    cam.elevation = -40
    cam.lookat[:] = np.array([5.0, 2.0, 0.0])

    print("Saving video to:", output_path)

    for frame in range(len(x_hist)):
        set_replay_frame(model, data, ids, x_hist, obs_hist, u_hist, frame)

        # Use free camera instead of XML camera
        renderer.update_scene(data, camera=cam)

        image = renderer.render()
        writer.append_data(image)

    writer.close()
    renderer.close()

    print("Video saved:", output_path)


def visualize_in_mujoco(x_hist, obs_hist, u_hist):
    """
    Live replay in MuJoCo viewer.
    """
    num_obs = obs_hist.shape[1]

    xml = build_mujoco_xml(num_obs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ids = get_body_ids(model, num_obs)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 14
        viewer.cam.azimuth = 10
        viewer.cam.elevation = -70
        viewer.cam.lookat[:] = np.array([8.0, 2.0, 0.0])

        frame = 0

        while viewer.is_running():
            set_replay_frame(model, data, ids, x_hist, obs_hist, u_hist, frame)

            viewer.sync()

            frame = (frame + 1) % len(x_hist)
            time.sleep(dt)



def plot_results(x_hist, obs_hist, u_hist):
    """
    Plot:
    1) Control inputs: acceleration and steering angle
    2) Vehicle x-y trajectory with obstacle trajectories
    """

    # Time vectors
    t_state = np.arange(len(x_hist)) * dt
    t_control = np.arange(len(u_hist)) * dt

    # Extract vehicle states
    x = x_hist[:, 0]
    y = x_hist[:, 1]
    psi = x_hist[:, 2]
    v = x_hist[:, 3]

    # Extract controls
    a = u_hist[:, 0]
    delta = u_hist[:, 1]

    # -----------------------------
    # Plot controls
    # -----------------------------
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

    # -----------------------------
    # Plot vehicle states
    # -----------------------------
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

    # -----------------------------
    # Plot x-y trajectory
    # -----------------------------
    plt.figure(figsize=(11, 6))

    # Vehicle trajectory
    plt.plot(x, y, linewidth=3, label="Vehicle trajectory")

    # Start and end points
    plt.scatter(x[0], y[0], s=80, marker="o", label="Start")
    plt.scatter(x[-1], y[-1], s=80, marker="x", label="End")

    # Reference lane y = 2
    plt.axhline(y=2.0, linestyle="--", linewidth=2, label="Reference path y = 2")

    # Obstacle trajectories
    num_obs = obs_hist.shape[1]

    for j in range(num_obs):
        ox = obs_hist[:, j, 0]
        oy = obs_hist[:, j, 1]

        plt.plot(ox, oy, linewidth=2, linestyle="--", label=f"Obstacle {j+1} path")
        plt.scatter(ox[0], oy[0], s=60, marker="o")
        plt.scatter(ox[-1], oy[-1], s=60, marker="x")

        # Plot final obstacle radius only, without margin
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


if __name__ == "__main__":
    x_hist, obs_hist, u_hist = run_nmpc_simulation()
    
    plot_results(x_hist, obs_hist, u_hist)
    if SAVE_VIDEO:
        save_mujoco_video(x_hist, obs_hist, u_hist, VIDEO_PATH)

    visualize_in_mujoco(x_hist, obs_hist, u_hist)
