import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

def quat_mult(q, r):
    """
    Vectorized quaternion multiplication.
    q and r should be numpy arrays of shape (4, N).
    """
    w = q[0, :] * r[0, :] - q[1, :] * r[1, :] - q[2, :] * r[2, :] - q[3, :] * r[3, :]
    x = q[0, :] * r[1, :] + q[1, :] * r[0, :] + q[2, :] * r[3, :] - q[3, :] * r[2, :]
    y = q[0, :] * r[2, :] - q[1, :] * r[3, :] + q[2, :] * r[0, :] + q[3, :] * r[1, :]
    z = q[0, :] * r[3, :] + q[1, :] * r[2, :] - q[2, :] * r[1, :] + q[3, :] * r[0, :]
    return np.vstack((w, x, y, z))

def normalize_quaternions(q):
    """Normalize each quaternion (columns of q) to have unit norm."""
    norms = np.linalg.norm(q, axis=0)
    return q / norms

def quat_inv(q):
    """Return the vectorized inverse of quaternion(s) q (assumed unit)."""
    q_inv = np.copy(q)
    q_inv[1:4, :] = -q_inv[1:4, :]
    return q_inv

def quat_to_axis_angle(q):
    """
    Convert a set of quaternions (shape (4, N)) to their axis-angle representation.
    Returns a (3, N) array where each column is the axis-angle vector.
    """
    q0 = np.clip(q[0, :], -1.0, 1.0)
    theta = 2 * np.arccos(q0)
    s = np.sqrt(1 - q0**2)
    axis = np.zeros_like(q[1:4, :])
    # Only perform division where s > 1e-8 to avoid divide-by-zero issues
    np.divide(q[1:4, :], s[np.newaxis, :], out=axis, where=(s[np.newaxis, :] > 1e-8))
    return axis * theta


# Convert digital IMU readings to analog values in physical units
# -------------------------------
def convert_imu_raw(raw_data, sensitivity, bias):
    """Convert raw IMU data to physical units."""
    return (raw_data - bias) * 3300 / (1023 * sensitivity)

# Generate sigma points for Unscented Kalman Filter (UKF)
# -------------------------------
# def generate_sigma_points(state, cov, process_noise):
#     """Generate sigma points from state and covariance matrix."""
#     sqrt_matrix = np.linalg.cholesky((cov + process_noise) * np.sqrt(6))
#     sigma_offsets = np.hstack((sqrt_matrix, -sqrt_matrix))
#     q_state = Quaternion(state[0, 0], state[1:4, 0])
#     q_state.normalize()
#     sigma_points = np.zeros((7, 12))
#     for i in range(12):
#         q_offset = Quaternion()
#         q_offset.from_axis_angle(sigma_offsets[0:3, i])
#         q_combined = q_state * q_offset
#         q_combined.normalize()
#         sigma_points[0:4, i] = q_combined.q
#         sigma_points[4:7, i] = state[4:7, 0] + sigma_offsets[3:6, i]
#     return sigma_points
def generate_sigma_points(state, cov, process_noise):
    """Generate sigma points from state and covariance matrix using vectorized operations."""
    sqrt_matrix = np.linalg.cholesky((cov + process_noise) * np.sqrt(6))
    sigma_offsets = np.hstack((sqrt_matrix, -sqrt_matrix))  # Shape: (6, 12)

    # Extract quaternion state and normalize
    q_state = Quaternion(state[0, 0], state[1:4, 0])
    q_state.normalize()

    # Convert sigma_offsets (first three rows) to quaternions
    angles = np.linalg.norm(sigma_offsets[0:3, :], axis=0)  # Compute axis-angle norms (12,)
    valid_axes = angles > 1e-8  # Avoid division by zero
    axes = np.zeros((3, 12))
    axes[:, valid_axes] = sigma_offsets[0:3, valid_axes] / angles[valid_axes]  # Normalize valid axes

    # Create quaternion offsets
    q_offsets = np.zeros((4, 12))  # Store quaternion representation
    q_offsets[0, :] = np.cos(angles / 2)
    q_offsets[1:4, :] = axes * np.sin(angles / 2)

    # Perform quaternion multiplication (vectorized)
    w = q_state.q[0] * q_offsets[0, :] - np.sum(q_state.q[1:4, np.newaxis] * q_offsets[1:4, :], axis=0)
    xyz = (
        q_state.q[0] * q_offsets[1:4, :]
        + q_offsets[0, :] * q_state.q[1:4, np.newaxis]
        + np.cross(q_state.q[1:4], q_offsets[1:4, :], axis=0)
    )

    # Store the quaternion components
    sigma_points = np.zeros((7, 12))
    sigma_points[0, :] = w
    sigma_points[1:4, :] = xyz
    sigma_points[4:7, :] = state[4:7, 0, np.newaxis] + sigma_offsets[3:6, :]

    return sigma_points
# Propagate sigma points through process model
# -------------------------------
# def propagate_sigma_points(sigma_points, delta_time):
#     """Propagate sigma points through system dynamics."""
#     for i in range(12):
#         q_rotation = Quaternion()
#         q_rotation.from_axis_angle(sigma_points[4:7, i] * delta_time)
#         q_current = Quaternion(sigma_points[0, i], sigma_points[1:4, i])
#         q_updated = q_current * q_rotation
#         q_updated.normalize()
#         sigma_points[0:4, i] = q_updated.q
#     return sigma_points

def propagate_sigma_points(sigma_points, delta_time):
    """Vectorized propagation of sigma points through system dynamics."""

    # Convert angular velocity * delta_time into axis-angle representations
    angles = np.linalg.norm(sigma_points[4:7, :] * delta_time, axis=0)
    axes = np.divide(sigma_points[4:7, :] * delta_time, angles, where=angles != 0)  # Avoid division by zero

    # Compute quaternion rotations
    q_rot = np.zeros((4, sigma_points.shape[1]))
    q_rot[0, :] = np.cos(angles / 2)
    q_rot[1:4, :] = axes * np.sin(angles / 2)

    # Extract current quaternions from sigma points
    q_current = sigma_points[0:4, :]

    # Perform quaternion multiplication (vectorized)
    w = q_current[0, :] * q_rot[0, :] - np.sum(q_current[1:4, :] * q_rot[1:4, :], axis=0)
    xyz = (
        q_current[0, :] * q_rot[1:4, :]
        + q_rot[0, :] * q_current[1:4, :]
        + np.cross(q_current[1:4, :].T, q_rot[1:4, :].T).T
    )

    # Normalize updated quaternions
    q_updated = np.vstack((w, xyz))
    q_updated /= np.linalg.norm(q_updated, axis=0)  # Normalize each quaternion

    # Update sigma points
    sigma_points[0:4, :] = q_updated
    return sigma_points
# Compute weighted mean quaternion from sigma points
# -------------------------------
# def compute_mean_quaternion(quat_sigma, initial_quat):
#     """Compute mean quaternion from sigma points."""
#     q_mean = Quaternion(initial_quat[0], initial_quat[1:4])
#     previous_error = 5
#     tolerance = 0.01 #1e-3 #before was 3 #2 is a good number
#     iteration = 0
#     mean_error = np.array([0.0, 0.0, 0.0])
#     while abs(np.linalg.norm(mean_error) - previous_error) > tolerance and iteration < 6: #before was 6
#         previous_error = np.linalg.norm(mean_error)
#         errors = np.zeros((3, 12))
#         for i in range(12):
#             q_i = Quaternion(quat_sigma[0, i], quat_sigma[1:4, i])
#             delta = q_i * q_mean.inv()
#             delta.normalize()
#             errors[:, i] = delta.axis_angle()
#         mean_error = np.mean(errors, axis=1)
#         q_correction = Quaternion()
#         q_correction.from_axis_angle(mean_error)
#         q_mean = q_correction * q_mean
#         q_mean.normalize()
#         iteration += 1
#     return q_mean

def compute_mean_quaternion(quat_sigma, initial_quat):
    """Vectorized computation of mean quaternion from sigma points."""
    # quat_sigma is shape (4, 12) and initial_quat is a 4-element vector.
    q_mean = initial_quat.reshape(4)
    q_mean = q_mean / np.linalg.norm(q_mean)
    previous_error = 5.0
    tolerance = 0.01
    iteration = 0
    mean_error = np.zeros(3)
    while abs(np.linalg.norm(mean_error) - previous_error) > tolerance and iteration < 6:
        previous_error = np.linalg.norm(mean_error)
        q_mean_rep = np.tile(q_mean.reshape(4, 1), (1, quat_sigma.shape[1]))
        inv_q_mean = quat_inv(q_mean_rep)
        delta = quat_mult(quat_sigma, inv_q_mean)
        delta = normalize_quaternions(delta)
        errors = quat_to_axis_angle(delta)  # shape (3, 12)
        mean_error = np.mean(errors, axis=1)
        angle = np.linalg.norm(mean_error)
        if angle > 1e-8:
            axis = mean_error / angle
        else:
            axis = np.array([1, 0, 0])
        q_correction = np.array([math.cos(angle / 2), *(axis * math.sin(angle / 2))]).reshape(4, 1)
        q_mean = quat_mult(q_correction, q_mean.reshape(4, 1)).reshape(4)
        q_mean = q_mean / np.linalg.norm(q_mean)
        iteration += 1
    q_mean_obj = Quaternion(q_mean[0], q_mean[1:4])
    q_mean_obj.q = q_mean
    return q_mean_obj

# Compute overall mean state from sigma points
# -------------------------------
def compute_mean_state(sigma_points, initial_quat):
    """Compute mean state including quaternion and angular velocity."""
    mean_state = np.zeros((7, 1))
    mean_state[4:7, 0] = np.mean(sigma_points[4:7, :], axis=1)
    mean_quat = compute_mean_quaternion(sigma_points[0:4, :], initial_quat)
    mean_state[0:4, 0] = mean_quat.q
    return mean_state

# Compute error covariance matrix from sigma points and mean
# -------------------------------
# def compute_sigma_errors(sigma_points, mean_state):
#     """Compute the sigma point errors relative to the mean."""
#     error_matrix = np.zeros((6, 12))
#     mean_quat = Quaternion(mean_state[0, 0], mean_state[1:4, 0])
#     mean_omega = mean_state[4:7, 0]
#     for i in range(12):
#         q_i = Quaternion(sigma_points[0, i], sigma_points[1:4, i])
#         delta_quat = q_i * mean_quat.inv()
#         delta_quat.normalize()
#         rot_vector = delta_quat.axis_angle()
#         omega_error = sigma_points[4:7, i] - mean_omega
#         error_matrix[:, i] = np.hstack((rot_vector, omega_error))
#     return error_matrix

def compute_sigma_errors(sigma_points, mean_state):
    """Vectorized computation of the sigma point errors relative to the mean."""
    mean_quat = mean_state[0:4, 0].reshape(4)
    mean_omega = mean_state[4:7, 0]
    q_sigma = sigma_points[0:4, :]
    mean_quat_rep = np.tile(mean_quat.reshape(4, 1), (1, q_sigma.shape[1]))
    inv_mean_quat = quat_inv(mean_quat_rep)
    delta = quat_mult(q_sigma, inv_mean_quat)
    delta = normalize_quaternions(delta)
    rot_vectors = quat_to_axis_angle(delta)  # (3, 12)
    omega_error = sigma_points[4:7, :] - mean_omega.reshape(3, 1)
    error_matrix = np.vstack((rot_vectors, omega_error))
    return error_matrix

# Predict measurements from sigma points
# -------------------------------
# def predict_measurements(sigma_points):
#     """Compute predicted measurements (accelerometer and gyro) from sigma points."""
#     predicted_measurements = np.zeros((6, 12))
#     gravity_quat = Quaternion(0.0, [0.0, 0.0, 9.8])
#     for i in range(12):
#         q_i = Quaternion(sigma_points[0, i], sigma_points[1:4, i])
#         rotated_gravity = q_i.inv() * gravity_quat * q_i
#         predicted_measurements[0:3, i] = rotated_gravity.vec()  # Accelerometer part
#         predicted_measurements[3:6, i] = sigma_points[4:7, i]  # Gyroscope (angular velocity)
#     return predicted_measurements

def predict_measurements(sigma_points):
    """Vectorized prediction of measurements (accelerometer and gyro) from sigma points."""
    # Create gravity quaternion as a (4,1) column vector.
    gravity_quat = np.array([0.0, 0.0, 0.0, 9.8]).reshape(4, 1)
    q_sigma = sigma_points[0:4, :]
    q_sigma_inv = quat_inv(q_sigma)
    gravity_rep = np.tile(gravity_quat, (1, q_sigma.shape[1]))
    temp = quat_mult(q_sigma_inv, gravity_rep)
    rotated = quat_mult(temp, q_sigma)
    predicted_measurements = np.zeros((6, q_sigma.shape[1]))
    predicted_measurements[0:3, :] = rotated[1:4, :]
    predicted_measurements[3:6, :] = sigma_points[4:7, :]
    return predicted_measurements


def estimate_rot(data_num):

    # Load data
    # ------------------------------
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    #vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    accel = imu['vals'][0:3, :]
    gyro = imu['vals'][3:6, :]
    T = np.shape(imu['ts'])[1]

    # Convert IMU raw data to physical units (m/s^2 and rad/s)
    # ------------------------------
    ax = -convert_imu_raw(accel[0, :], 25.413113426714844, 510.78)
    ay = -convert_imu_raw(accel[1, :], 25.413113426714844, 501.0) #501
    az = convert_imu_raw(accel[2, :], 25.413113426714844, 503.75)
    gyro = gyro[[1, 2, 0], :]
    gx = convert_imu_raw(gyro[0, :], 200.96194618264164, 373.63) #200.97156746197862 #decreases 200.96194618264164
    gy = convert_imu_raw(gyro[1, :], 201.634982427985, 375.2) #201.63112899772713 #decreases increase points
    gz = convert_imu_raw(gyro[2, :], 201.78139222013317, 369.66) #202.105372905874354

    # Time difference array
    # ------------------------------
    delta_t = imu['ts'][0, 1:] - imu['ts'][0, :-1]
    delta_t = np.hstack((0, delta_t))

    # Initialize UKF state and parameters
    # ------------------------------
    x_k_1 = np.zeros((7, 1))
    x_k_1[0, 0] = 1  # Initial quaternion (identity)
    P_k_1 = np.zeros((6, 6))  # Start with no uncertainty (fully confident in the initial state)
    Q = np.diag([np.pi /2.1, np.pi / 2, np.pi / 1.8, np.pi / 2.5, np.pi / 2.5, np.pi / 2.5])
    R = np.diag([np.pi / 0.25, np.pi / 0.6, np.pi / 0.6, np.pi / 0.6, np.pi / 0.6, np.pi / 0.6])
    roll = np.zeros(T)
    pitch = np.zeros(T)
    yaw = np.zeros(T)

    P_history = np.zeros((6, 6, T))  # Store covariance matrix at each time step

    x_k_1_history = np.zeros((7, T))  # Store quaternion and angular velocity

    # UKF Loop
    # ------------------------------

    for i in range(T):
        # Generate sigma points
        sigma_points = generate_sigma_points(x_k_1, P_k_1, Q)
        # Propagate sigma points
        propagated_sigmas = propagate_sigma_points(sigma_points, delta_t[i])
        # Compute predicted mean state
        x_k_bar = compute_mean_state(propagated_sigmas, sigma_points[0:4, 0])
        # Compute sigma deviations
        sigma_errors = compute_sigma_errors(propagated_sigmas, x_k_bar)
        # Predict covariance
        P_k_bar = (sigma_errors @ sigma_errors.T) / 12.0
        # Predict measurements
        Z_sigma = predict_measurements(propagated_sigmas)
        # Measurement mean
        Z_mean = np.mean(Z_sigma, axis=1).reshape(6, 1)
        # Observed measurements
        Z_obs = np.zeros((6, 1))
        Z_obs[0:3, 0] = np.array([ax[i], ay[i], az[i]])
        Z_obs[3:6, 0] = np.array([gx[i], gy[i], gz[i]])
        # Innovation covariance
        Z_dev = Z_sigma - Z_mean
        P_zz = (Z_dev @ Z_dev.T) / 12.0
        P_vv = P_zz + R
        # Cross-covariance
        P_xz = (sigma_errors @ Z_dev.T) / 12.0
        # Kalman gain
        K = P_xz @ np.linalg.inv(P_vv)
        # Update covariance
        P_k = P_k_bar - K @ P_vv @ K.T
        # Innovation term
        innovation = K @ (Z_obs - Z_mean)
        # Update angular velocity
        x_k_bar[4:7, 0] += innovation[3:6, 0]
        # Update quaternion
        q_innovation = Quaternion()
        q_innovation.from_axis_angle(innovation[0:3, 0])
        q_mean = Quaternion(x_k_bar[0, 0], x_k_bar[1:4, 0])
        q_updated = q_innovation * q_mean
        q_updated.normalize()
        x_k_bar[0:4, 0] = q_updated.q
        # Save updated state
        x_k_1 = x_k_bar
        #print(P_k)
        P_k_1 = P_k
        #print("P_k_1", P_k_1)

        x_k_1_history[:, i] = x_k_1[:, 0]  # Save full state
        P_history[:, :, i] = P_k_1  # <---- ADD THIS LINE to save covariance

        # Extract Euler angles (roll, pitch, yaw)
        # ------------------------------
        euler_angles = q_updated.euler_angles()
        roll[i] = euler_angles[0]
        pitch[i] = euler_angles[1]
        yaw[i] = euler_angles[2]

    # Return final results
    # ------------------------------

    # ---------- Quaternion Components (4 subplots) ----------
    # labels_q = ['q_w', 'q_x', 'q_y', 'q_z']
    #
    # fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    #
    # for i in range(4):
    #     mean = x_k_1_history[i, :]
    #     std_dev = np.sqrt(P_history[i, i, :])
    #     axs[i].plot(mean, label=f'{labels_q[i]} (mean)')
    #     axs[i].fill_between(range(len(mean)), mean - std_dev, mean + std_dev, alpha=0.3, label='Std Dev')
    #     axs[i].legend()
    #     axs[i].grid(True)
    #     axs[i].set_ylabel(labels_q[i])
    #
    # axs[-1].set_xlabel('Time Step')
    # fig.suptitle('Quaternion Components with Uncertainty (Shaded Std Dev)')
    # plt.show()
    #
    # # ---------- Angular Velocity Components (with uncertainty) ----------
    # labels_w = ['ω_x', 'ω_y', 'ω_z']
    # fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # for i in range(3):
    #     mean = x_k_1_history[4 + i, :]
    #     std_dev = np.sqrt(P_history[3 + i, 3 + i, :])  # Correct part of covariance
    #     axs[i].plot(mean, label=f'{labels_w[i]} (mean)')
    #     axs[i].fill_between(range(len(mean)), mean - std_dev, mean + std_dev, alpha=0.3, label='Std Dev')
    #     axs[i].legend()
    #     axs[i].grid(True)
    #     axs[i].set_ylabel(labels_w[i] + ' (rad/s)')
    # axs[-1].set_xlabel('Time Step')
    # fig.suptitle('Angular Velocity Components with Uncertainty (Shaded Std Dev)')
    # plt.show()

    return roll, pitch, yaw

##################QUALITY CHECK##############################

roll, pitch, yaw = estimate_rot(2)

plt.figure()
plt.plot(roll, label='Roll')
plt.plot(pitch, label='Pitch')
plt.plot(yaw, label='Yaw')
plt.legend()
plt.title('Estimated Roll, Pitch, Yaw over Time')
plt.xlabel('Time Step')
plt.ylabel('Angle (radians)')
plt.grid(True)
plt.show()

data_num = 2
imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
vicon_rot = vicon['rots']  # This is a 3x3xT rotation matrix over time
T = np.shape(imu['ts'])[1]-100
# Arrays to store ground truth
roll_gt = np.zeros(T)
pitch_gt = np.zeros(T)
yaw_gt = np.zeros(T)

def rot_to_euler(R):
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw)"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x, y, z  # roll, pitch, yaw

for i in range(T):
    R = vicon_rot[:, :, i]
    roll_gt[i], pitch_gt[i], yaw_gt[i] = rot_to_euler(R)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Roll
plt.subplot(3, 1, 1)
plt.plot(roll, label='Estimated Roll')
plt.plot(roll_gt, label='Ground Truth Roll', linestyle='dashed')
plt.legend()
plt.ylabel('Roll (rad)')
plt.grid(True)

# Pitch
plt.subplot(3, 1, 2)
plt.plot(pitch, label='Estimated Pitch')
plt.plot(pitch_gt, label='Ground Truth Pitch', linestyle='dashed')
plt.legend()
plt.ylabel('Pitch (rad)')
plt.grid(True)

# Yaw
plt.subplot(3, 1, 3)
# plt.plot(yaw, label='Estimated Yaw')
plt.plot(yaw, label='Estimated Yaw')
plt.plot(yaw_gt, label='Ground Truth Yaw', linestyle='dashed')
plt.legend()
plt.ylabel('Yaw (rad)')
plt.xlabel('Time Step')
plt.grid(True)
plt.suptitle('UKF Estimated vs. Vicon Ground Truth (Roll, Pitch, Yaw)')
plt.show()
#
# # ------------------------------
# # 3. Plot Gyroscope readings (rad/s) as separate subplots
# # ------------------------------
# gyro = imu['vals'][3:6, :]
# gyro = gyro[[1, 2, 0], :]  # reorder
# gx = convert_imu_raw(gyro[0, :], 200.96194618264164, 373.63)
# gy = convert_imu_raw(gyro[1, :], 201.634982427985, 375.2)
# gz = convert_imu_raw(gyro[2, :], 201.78139222013317, 369.66)
#
# gyro_labels = ['Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']
# gyro_data = [gx, gy, gz]
#
# fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#
# for i in range(3):
#     axs[i].plot(gyro_data[i], label=gyro_labels[i])
#     axs[i].legend()
#     axs[i].grid(True)
#     axs[i].set_ylabel('Rad/s')
#
# axs[-1].set_xlabel('Time Step')
# fig.suptitle('Gyroscope Readings (rad/s)')
# plt.show()
#
# # ------------------------------
# # 4. Plot Vicon Quaternion (converted from rotation matrix) as separate subplots
# # ------------------------------
# from scipy.spatial.transform import Rotation as R
#
# vicon_quat = np.zeros((4, T))  # (w, x, y, z)
# for i in range(T):
#     rot = R.from_matrix(vicon_rot[:, :, i])
#     q = rot.as_quat()  # (x, y, z, w)
#     vicon_quat[:, i] = np.array([q[3], q[0], q[1], q[2]])  # (w, x, y, z)
#
# labels_q = ['q_w', 'q_x', 'q_y', 'q_z']
#
# fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
#
# for i in range(4):
#     axs[i].plot(vicon_quat[i, :], label=f'Vicon {labels_q[i]}')
#     axs[i].legend()
#     axs[i].grid(True)
#     axs[i].set_ylabel(labels_q[i])
#
# axs[-1].set_xlabel('Time Step')
# fig.suptitle('Vicon Quaternion Components over Time')
# plt.show()
