
##############################################################THIRD VER
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ------------------------------
# Load Data
# ------------------------------
data_num = 3  # You can change this as needed
imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')

accel = imu['vals'][0:3, :]
gyro = imu['vals'][3:6, :]  # raw gyroscope data
gyro = gyro[[1, 2, 0], :]  # reordering axes
rots = vicon['rots']  # (3, 3, T)
T_vicon = rots.shape[2]
dt = 0.01  # Assuming 100 Hz

# ------------------------------
# Convert Vicon rotations to Euler angles (roll, pitch, yaw)
# ------------------------------
vicon_euler = np.zeros((3, T_vicon))
for t in range(T_vicon):
    vicon_euler[:, t] = R.from_matrix(rots[:, :, t]).as_euler('xyz', degrees=False)

# ------------------------------
# Step 1: Estimate Gyroscope Bias using first N static samples
# ------------------------------
N_bias = 100  # Number of samples assumed stationary
bias = np.mean(gyro[:, 0:N_bias], axis=1)  # Bias for each axis
print("Estimated Gyroscope Biases (raw units):", bias)

# ------------------------------
# Step 2: Debias Gyroscope
# ------------------------------
gyro_debiased = gyro - bias.reshape(3, 1)  # Subtract bias

# ------------------------------
# Step 3: Integrate Gyroscope to get uncorrected orientation
# ------------------------------
gyro_integrated = np.cumsum(gyro_debiased * dt, axis=1)  # Integrated gyro

# ------------------------------
# Step 4: Least Squares Calibration (Alpha scaling) per axis
# ------------------------------
# Define custom fitting ranges for each axis
range_x = (1478, 1866) #  #fixed, take first map (3478, 3866) 200.96194618264164
range_y = (1513, 1600) # #fixed, take first map ((2513, 3600)) 201.634982427985
range_z = (981, 1956)   #fixed, take third map (980, 1956) 201.78139222013317


# ------------------------------
# Roll (X-axis)
# ------------------------------
gyro_x = gyro_integrated[0, :T_vicon]  # Integrated Gyro X
vicon_x = vicon_euler[0, :]

start_x, end_x = range_x
gyro_x_window = gyro_x[start_x:end_x]
vicon_x_window = vicon_x[start_x:end_x]

# # Least squares

# ------------------------------
# Roll (X-axis)
# ------------------------------
A_x = np.vstack([gyro_x_window, np.ones_like(gyro_x_window)]).T
x_x, _, _, _ = np.linalg.lstsq(A_x, vicon_x_window, rcond=None)
alpha_x, beta_x = x_x

# ------------------------------
# Pitch (Y-axis)
# ------------------------------
gyro_y = gyro_integrated[1, :T_vicon]
vicon_y = vicon_euler[1, :]

start_y, end_y = range_y
gyro_y_window = gyro_y[start_y:end_y]
vicon_y_window = vicon_y[start_y:end_y]

# A_y = gyro_y_window.reshape(-1, 1)
# alpha_y, _, _, _ = np.linalg.lstsq(A_y, vicon_y_window, rcond=None)

# ------------------------------
# Pitch (Y-axis)
# ------------------------------
A_y = np.vstack([gyro_y_window, np.ones_like(gyro_y_window)]).T
x_y, _, _, _ = np.linalg.lstsq(A_y, vicon_y_window, rcond=None)
alpha_y, beta_y = x_y

# ------------------------------
# Yaw (Z-axis)
# ------------------------------
gyro_z = gyro_integrated[2, :T_vicon]
vicon_z = vicon_euler[2, :]

start_z, end_z = range_z
gyro_z_window = gyro_z[start_z:end_z]
vicon_z_window = vicon_z[start_z:end_z]
A_z = np.vstack([gyro_z_window, np.ones_like(gyro_z_window)]).T
x_z, _, _, _ = np.linalg.lstsq(A_z, vicon_z_window, rcond=None)
alpha_z, beta_z = x_z

# ------------------------------
# Print Estimated Alphas
# ------------------------------
print("\nEstimated Alpha (scale) factors:")
print(f"Roll (X): {3300/(1023*alpha_x)}")
print(f"Pitch (Y): {3300/(1023*alpha_y)}")
print(f"Yaw (Z): {3300/(1023*alpha_z)}")

gyro_x_calibrated = alpha_x * gyro_integrated[0, :] + beta_x
gyro_y_calibrated = alpha_y * gyro_integrated[1, :] + beta_y
gyro_z_calibrated = alpha_z * gyro_integrated[2, :] + beta_z

# ------------------------------
# Plotting Results
# ------------------------------
plt.figure(figsize=(15, 8))

# Roll (X)
plt.subplot(3, 1, 1)
plt.plot(vicon_x, label='Vicon Roll (X)', linewidth=2)
plt.plot(gyro_x_calibrated, label='Calibrated Gyro Roll (X)', linestyle='-.')
plt.legend()
plt.ylabel('Radians')
plt.title('Roll (X)')

# Pitch (Y)
plt.subplot(3, 1, 2)
plt.plot(vicon_y, label='Vicon Pitch (Y)', linewidth=2)
plt.plot(gyro_y_calibrated, label='Calibrated Gyro Pitch (Y)', linestyle='-.')
plt.legend()
plt.ylabel('Radians')
plt.title('Pitch (Y)')

# Yaw (Z)
plt.subplot(3, 1, 3)
plt.plot(vicon_z, label='Vicon Yaw (Z)', linewidth=2)
plt.plot(gyro_z_calibrated, label='Calibrated Gyro Yaw (Z)', linestyle='-.')
plt.legend()
plt.ylabel('Radians')
plt.title('Yaw (Z)')

plt.xlabel('Timestep')
plt.tight_layout()
plt.show()

