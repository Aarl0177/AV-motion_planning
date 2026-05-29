import numpy as np

# NMPC settings
N = 60
Nt = 160
T = 16.0
dt = T / Nt
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
margin = 0.30
safe_radius = obs_radius + vehicle_radius + margin

# Video settings
SAVE_VIDEO = True
VIDEO_PATH = "stage2_mujoco_car_lidar_simulation.mp4"
VIDEO_FPS = 10
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720