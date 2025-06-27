# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import os, sys, pickle, math
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from load_data import load_kitti_lidar_data, load_kitti_poses, load_kitti_calib
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    def __init__(s, resolution=0.5):
        s.resolution = resolution
        s.xmin, s.xmax = -700, 700
        s.zmin, s.zmax = -500, 900
        # s.xmin, s.xmax = -400, 1100
        # s.zmin, s.zmax = -300, 1200

        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szz = int(np.ceil((s.zmax - s.zmin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szz), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds,
        # and similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xz(s, x, z):
        """
        x and z can be 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/z go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        x_arr = np.atleast_1d(x)
        z_arr = np.atleast_1d(z)

        # Make sure these match your visualization expectations
        cols = np.floor((x_arr - s.xmin) / s.resolution).astype(int)
        rows = np.floor((z_arr - s.zmin) / s.resolution).astype(int)

        cols = np.clip(cols, 0, s.szx - 1)
        rows = np.clip(rows, 0, s.szz - 1)

        return np.vstack((rows, cols))

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.5, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

        # dynamics noise for the state (x, z, yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar_dir = src_dir + f'odometry/{s.idx}/velodyne/'
        s.poses = load_kitti_poses(src_dir + f'poses/{s.idx}.txt')
        s.lidar_files = sorted(os.listdir(src_dir + f'odometry/{s.idx}/velodyne/'))
        s.calib = load_kitti_calib(src_dir + f'calib/{s.idx}/calib.txt')
        # Convert the calibration matrix (3x4) to 4x4 and store it.
        Tr = s.calib  # s.calib is already a 3x4 matrix
        s.calib_mat = np.vstack((Tr, np.array([0, 0, 0, 1])))

    def init_particles(s, n=100, p=None, w=None):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n))
        s.w = deepcopy(w) if w is not None else np.ones(n) / n

    @staticmethod
    def stratified_resampling(p, w):
        """
        Resampling step of the particle filter.
        """
        n = len(w)
        cumulative_sum = np.cumsum(w)
        cumulative_sum[-1] = 1.0  # ensure the last value is exactly one
        indexes = np.zeros(n, dtype=int)
        r = np.random.uniform(0, 1.0 / n)
        positions = r + np.arange(n) / n
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        new_p = p[:, indexes]
        new_w = np.ones(n) / n
        return new_p, new_w




    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())


    def lidar2world(s, p, points):
        """
        Transforms LiDAR points to world coordinates.

        The particle state p is now interpreted as [x, z, theta], where:
        - p[0]: x translation
        - p[1]: z translation
        - p[2]: rotation in the x-z plane

        The input 'points' is an (N, 3) array of LiDAR points in xyz.
        """
        N = points.shape[0]
        pts_hom = np.hstack((points, np.ones((N, 1))))  # (N, 4)

        # Apply calibration transform (velodyne -> camera)
        pts_robot = (s.calib_mat @ pts_hom.T).T  # (N, 4)

        # For a correct top-down view, we use X as forward and Z as right
        # (This matches standard robotics convention for ground vehicles)
        pts_2d = pts_robot[:, [0, 2]]  # (N, 2)

        # Create SE(2) transform from particle pose [x, z, theta]
        # Make sure the rotation matrix matches your coordinate convention
        cos_theta, sin_theta = np.cos(p[2]), np.sin(p[2])
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        t = np.array([p[0], p[1]])

        # Apply transform: rotation then translation
        pts_world = (R @ pts_2d.T).T + t

        return pts_world

    # def get_control(s, t):
    #     """
    #     Use the pose at time t and t-1 to calculate what control the robot could have taken
    #     at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
    #     assume that this is the same control that the robot will take in the function dynamics_step
    #     below at time t, to go to time t-1. need to use the smart_minus_2d
    #     function to get the difference of the two poses and we will simply
    #     set this to be the control.
    #     Extracts control in the state space [x, z, rotation] from consecutive poses.
    #     [x, z, theta]
    #     theta is the rotation around the Y-axis
    #           | cos  0  -sin |
    #     R_y = |  0   1    0  |
    #           |+sin  0   cos |
    #     R31 = +sin
    #     R11 =  cos
    #     yaw = atan2(R_31, R_11)
    #     """
    #     if t == 0:
    #         return np.zeros(3)
    #
    #     #### TODO: XXXXXXXXXXX
    #     raise NotImplementedError

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d
        function to get the difference of the two poses and we will simply
        set this to be the control.
        Extracts control in the state space [x, z, rotation] from consecutive poses.
        [x, z, theta]
        theta is the rotation around the Y-axis
              | cos  0  -sin |
        R_y = |  0   1    0  |
              |+sin  0   cos |
        R31 = +sin
        R11 =  cos
        yaw = atan2(R_31, R_11)
        """
        if t == 0:
            return np.zeros(3)

        prev = s.poses[t - 1]
        curr = s.poses[t]

        # Extract poses with consistent coordinate interpretation
        prev_pose = np.array([
            prev[0, 3],  # x
            prev[2, 3],  # z (used as y in 2D)
            np.arctan2(-prev[2, 0], prev[0, 0])  # yaw from rotation matrix
        ])

        curr_pose = np.array([
            curr[0, 3],  # x
            curr[2, 3],  # z (used as y in 2D)
            np.arctan2(-curr[2, 0], curr[0, 0])  # yaw from rotation matrix
        ])

        # Calculate control (delta position + orientation)
        control = smart_minus_2d(curr_pose, prev_pose)
        return control

    def dynamics_step(s, t):
        """
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter
        """
        control = s.get_control(t)
        for i in range(s.n):
            s.p[:, i] = smart_plus_2d(s.p[:, i], control)
            noise = np.random.multivariate_normal(np.zeros(3), s.Q)
            s.p[:, i] += noise




    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        max_logp = np.max(obs_logp)
        new_w = w * np.exp(obs_logp - max_logp)
        new_w = new_w / (np.sum(new_w) + 1e-9)
        return new_w


    # def observation_step(s, t):
    #     """
    #     This function does the following things
    #         1. updates the particles using the LiDAR observations
    #         2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
    #     you can also store a thresholded version of the map here for plotting later
    #     """
    #
    #     #### TODO: XXXXXXXXXXX
    #     raise NotImplementedError

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        you can also store a thresholded version of the map here for plotting later
        """
        # Load the LiDAR scan for time t.
        lidar_filename = os.path.join(s.lidar_dir, s.lidar_files[t])
        points = load_kitti_lidar_data(lidar_filename)[:, :3]  # use only xyz
        points = clean_point_cloud(load_kitti_lidar_data(lidar_filename))[:, :3]

        obs_logp = np.zeros(s.n)
        for i in range(s.n):
            pts_world = s.lidar2world(s.p[:, i], points)
            # Convert world coordinates to grid indices.
            inds = s.map.grid_cell_from_xz(pts_world[:, 0], pts_world[:, 1])
            rows = inds[0, :]
            cols = inds[1, :]
            # Sum the current log_odds in those cells as the observation log-probability.
            obs_logp[i] = np.sum(s.map.log_odds[rows, cols])
        # Update particle weights.
        s.w = s.update_weights(s.w, obs_logp)

        # Mapping update: use the best particle (largest weight).
        best_idx = np.argmax(s.w)
        best_particle = s.p[:, best_idx]
        pts_world_best = s.lidar2world(best_particle, points)
        inds_best = s.map.grid_cell_from_xz(pts_world_best[:, 0], pts_world_best[:, 1])
        rows_best = inds_best[0, :]
        cols_best = inds_best[1, :]
        # For each cell hit by LiDAR in the best particle's scan, increase log_odds.
        s.map.log_odds[rows_best, cols_best] += s.lidar_log_odds_occ

        # Recompute the binarized map: mark a cell as occupied if log_odds >= threshold.
        s.map.cells = (s.map.log_odds >= s.map.log_odds_thresh).astype(np.int8)


    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1 / np.sum(s.w ** 2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e / s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')

