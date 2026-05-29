import time
import numpy as np
import mujoco
import mujoco.viewer
import imageio.v2 as imageio

from config import (
    dt,
    obs_radius,
    safe_radius,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    VIDEO_PATH,
)


# --------------------------------------------------
# Quaternion helper functions
# --------------------------------------------------

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


# --------------------------------------------------
# MuJoCo XML scene
# --------------------------------------------------

def build_mujoco_xml(num_obs):
    """
    Build the MuJoCo XML scene.

    The car is visual/kinematic:
    its position and yaw are updated from the bicycle model.
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
                  size="{obs_radius} 0.015"
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
    </asset>

    <worldbody>

        <light pos="0 0 8"
               dir="0 0 -1"
               diffuse="1 1 1"/>

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

        <!-- Visual car body -->
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

            <!-- Two visual LiDAR beams -->
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


# --------------------------------------------------
# MuJoCo body ID helper
# --------------------------------------------------

def get_body_ids(model, num_obs):
    """
    Store MuJoCo body IDs so we do not repeatedly search by name.
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


# --------------------------------------------------
# Replay frame update
# --------------------------------------------------

def set_replay_frame(model, data, ids, x_hist, obs_hist, u_hist, frame):
    """
    Update one MuJoCo frame using stored simulation history.
    """

    num_obs = obs_hist.shape[1]

    # Vehicle state: [x, y, psi, v]
    x_pos = x_hist[frame, 0]
    y_pos = x_hist[frame, 1]
    psi = x_hist[frame, 2]

    # Control input: [a, delta]
    if len(u_hist) == 0:
        delta = 0.0
    elif frame < len(u_hist):
        delta = u_hist[frame, 1]
    else:
        delta = u_hist[-1, 1]

    # Move vehicle body
    model.body_pos[ids["vehicle"]] = np.array([x_pos, y_pos, 0.22])
    model.body_quat[ids["vehicle"]] = yaw_to_quat_z(psi)

    # Wheel orientation
    wheel_base_quat = quat_x(np.pi / 2)

    # Front wheels steer according to delta
    front_wheel_quat = quat_mul(yaw_to_quat_z(delta), wheel_base_quat)

    model.body_quat[ids["front_left_wheel"]] = front_wheel_quat
    model.body_quat[ids["front_right_wheel"]] = front_wheel_quat

    # Rear wheels stay straight
    model.body_quat[ids["rear_left_wheel"]] = wheel_base_quat
    model.body_quat[ids["rear_right_wheel"]] = wheel_base_quat

    # Move obstacles and circles
    for j in range(num_obs):
        ox = obs_hist[frame, j, 0]
        oy = obs_hist[frame, j, 1]

        model.body_pos[ids["obs"][j]] = np.array([ox, oy, 0.30])
        model.body_pos[ids["safe"][j]] = np.array([ox, oy, 0.015])

    mujoco.mj_forward(model, data)


# --------------------------------------------------
# Save MuJoCo video
# --------------------------------------------------

def save_mujoco_video(x_hist, obs_hist, u_hist, output_path=VIDEO_PATH):
    """
    Save MuJoCo replay as an MP4 video.

    This camera follows the vehicle so the car is visible from
    the beginning to the end of the video.
    """

    num_obs = obs_hist.shape[1]

    xml = build_mujoco_xml(num_obs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ids = get_body_ids(model, num_obs)

    renderer = mujoco.Renderer(
        model,
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH
    )

    writer = imageio.get_writer(
        output_path,
        fps=VIDEO_FPS,
        macro_block_size=1
    )

    # Free camera for saved video
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    cam.distance = 14
    cam.azimuth = 10
    cam.elevation = -70
    cam.lookat[:] = np.array([0.0, 0.0, 0.0])

    print("Saving video to:", output_path)

    for frame in range(len(x_hist)):
        set_replay_frame(model, data, ids, x_hist, obs_hist, u_hist, frame)

        # Follow vehicle
        x_pos = x_hist[frame, 0]
        y_pos = x_hist[frame, 1]

        cam.lookat[:] = np.array([x_pos + 2.0, y_pos, 0.0])

        renderer.update_scene(data, camera=cam)
        image = renderer.render()
        writer.append_data(image)

    writer.close()
    renderer.close()

    print("Video saved:", output_path)


# --------------------------------------------------
# Live MuJoCo viewer
# --------------------------------------------------

def visualize_in_mujoco(x_hist, obs_hist, u_hist):
    """
    Open live MuJoCo viewer and replay the simulation.
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
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])

        frame = 0

        while viewer.is_running():
            set_replay_frame(model, data, ids, x_hist, obs_hist, u_hist, frame)

            # Follow vehicle in live viewer too
            x_pos = x_hist[frame, 0]
            y_pos = x_hist[frame, 1]

            viewer.cam.lookat[:] = np.array([x_pos + 2.0, y_pos, 0.0])

            viewer.sync()

            frame = (frame + 1) % len(x_hist)
            time.sleep(dt)