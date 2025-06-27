import numpy as np
import matplotlib.pyplot as plt

# Given variables
R = 0.5  # Tuned process noise covariance
Q = 0.1  # Tuned measurement noise covariance
a = -1

num_observations = 100  # Increased number of observations

# QUESTION A
observation_dataset = []
np.random.seed(1)
initial_state_x = np.random.normal(1, 2)
current_state_x = a * initial_state_x + np.random.normal(0, 1)
previous_state_x = current_state_x


for iteration_index in range(num_observations):
    next_state_x = a * previous_state_x + np.random.normal(0, 1)
    observation_y = np.sqrt(previous_state_x ** 2 + 1) + np.random.normal(0, np.sqrt(Q))
    observation_dataset.append(observation_y)
    previous_state_x = next_state_x

# QUESTION B
previous_covariance_matrix = np.diag([2, 1])
process_noise_beta = 0.1
estimated_mean_a = -5
estimated_mean_x = 1

true_a_values = []
lower_bound_estimate = []
upper_bound_estimate = []
estimated_mean_values = []

for iteration_index in range(num_observations):  # Loop for more observations
    # Propagation step
    state_transition_matrix_A = np.array([[estimated_mean_a, estimated_mean_x], [0, 1]])
    mean_vector = np.array([[estimated_mean_a * estimated_mean_x], [estimated_mean_a]])
    covariance_matrix = (state_transition_matrix_A @ previous_covariance_matrix @ state_transition_matrix_A.T
                         + np.diag([R, process_noise_beta]))

    estimated_mean_x = mean_vector[0, 0]
    estimated_mean_a = mean_vector[1, 0]

    # Incorporating observation
    observation_matrix_C = np.zeros((1, 2))
    if abs(estimated_mean_x) < 1e-6:
        observation_matrix_C[0, 0] = 0  # Prevent division instability
    else:
        observation_matrix_C[0, 0] = estimated_mean_x / np.sqrt(estimated_mean_x ** 2 + 1)

    # Compute Kalman Gain
    kalman_gain = (covariance_matrix @ observation_matrix_C.T @
                   np.linalg.inv(observation_matrix_C @ covariance_matrix @ observation_matrix_C.T + Q + 1e-6 * np.eye(1)))

    updated_mean_vector = mean_vector + kalman_gain * (observation_dataset[iteration_index] - np.sqrt(estimated_mean_x ** 2 + 1))
    updated_covariance_matrix = (np.eye(2, 2) - kalman_gain @ observation_matrix_C) @ covariance_matrix + 1e-6 * np.eye(2)

    previous_covariance_matrix = updated_covariance_matrix
    estimated_mean_x = updated_mean_vector[0, 0]
    estimated_mean_a = updated_mean_vector[1, 0]

    true_a_values.append(-1)
    lower_bound_estimate.append(estimated_mean_a - np.sqrt(updated_covariance_matrix[1, 1]))
    upper_bound_estimate.append(estimated_mean_a + np.sqrt(updated_covariance_matrix[1, 1]))
    estimated_mean_values.append(estimated_mean_a)

time_steps_k = np.arange(1, num_observations + 1)
plt.plot(time_steps_k, true_a_values, label="True value (a = -1)", linestyle='dotted', color='black')
plt.plot(time_steps_k, estimated_mean_values, label="Estimated mean value", color='blue')
plt.xlabel("Time steps k")
plt.ylabel("System parameter a")
plt.legend()
plt.fill_between(time_steps_k, lower_bound_estimate, upper_bound_estimate, alpha=0.2, color='red')
plt.title("EKF Estimation of a with Uncertainty Bounds")
plt.show()
