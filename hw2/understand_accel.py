from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Quaternion

def transform(raw_data, bias, sensitivity):
    result = (raw_data - bias) * 3300 / (1023 * sensitivity)
    return result

sample_id = 1
imu_data = io.loadmat('imu/imuRaw' + str(sample_id) + '.mat')
vicon_data = io.loadmat('vicon/viconRot' + str(sample_id) + '.mat')
accelerometer = imu_data['vals'][0:3, :]
gyroscope = imu_data['vals'][3:6, :]
T = np.shape(imu_data['ts'])[1]
print(T)
time_steps = T - 500

bias_x = np.mean(accelerometer[0, 0:50])
bias_y = np.mean(accelerometer[1, 0:50])

print("Bias X:", bias_x)
print("Bias Y:", bias_y)
accel_z_values = []

quat_handler = Quaternion()
for idx in range(1000,time_steps):
    rotation_matrix = vicon_data['rots'][:, :, idx].reshape(3, 3)
    quaternion = quat_handler.from_rotm(rotation_matrix)
    euler_angles = quat_handler.euler_angles()
    if (np.pi / 2 - 0.01) <= euler_angles[0] <= (np.pi / 2 + 0.01):
        accel_z_values.append(accelerometer[2, idx])

bias_z = np.mean(accel_z_values)
print("Bias Z:", bias_z)

raw_z_values = accelerometer[2, 0:time_steps]
sensitivity_factor = np.mean((raw_z_values - bias_z) * 3300 / (1023*9.81))
print("Sensitivity Factor:", sensitivity_factor)

roll_vals, pitch_vals, yaw_vals = [], [], []
for idx in range(vicon_data['rots'].shape[-1]):
    rotation_matrix = vicon_data['rots'][:, :, idx].reshape(3, 3)
    quaternion = quat_handler.from_rotm(rotation_matrix)
    euler_angles = quat_handler.euler_angles()
    roll_vals.append(euler_angles[0])
    pitch_vals.append(euler_angles[1])
    yaw_vals.append(euler_angles[2])



converted_x = transform(accelerometer[0, :], bias_x, sensitivity_factor)
converted_y = transform(accelerometer[1, :], bias_y, sensitivity_factor)
converted_z = transform(accelerometer[2, :], bias_z, sensitivity_factor)

calculated_roll = np.arctan(-converted_y / converted_z)
calculated_pitch = np.arctan(converted_x / np.sqrt(converted_y ** 2 + converted_z ** 2))



# Plot both roll and pitch using subplots
plt.figure(figsize=(12, 8))

# Plot roll comparison
plt.subplot(2, 1, 1)
plt.plot(roll_vals, label="Vicon Roll")
plt.plot(calculated_roll, label="Calculated Roll")
plt.title("Roll Comparison")
plt.xlabel("Time Step")
plt.ylabel("Roll (radians)")
plt.legend()
plt.grid(True)

# Plot pitch comparison
plt.subplot(2, 1, 2)
plt.plot(pitch_vals, label="Vicon Pitch")
plt.plot(calculated_pitch, label="Calculated Pitch")
plt.title("Pitch Comparison")
plt.xlabel("Time Step")
plt.ylabel("Pitch (radians)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()