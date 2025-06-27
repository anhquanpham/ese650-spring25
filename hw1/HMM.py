import numpy as np
#import pandas as pd ##UNCOMMENT THIS LINE TO DISPLAY RESULTS

class HMM():
    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution
        self.num_states = Transition.shape[0]
        self.num_observations = len(Observations)

        # Mapping from state indices to city names #
        #self.state_to_city = {0: 'LA', 1: 'NY'} #UNCOMMENT THIS LINE TO RUN RESULTS

    def forward(self):
        """
        alpha = ...
        """
        alpha = np.zeros((self.num_observations, self.num_states))
        alpha[0] = self.Initial_distribution * self.Emission[:, self.Observations[0]]
        for t in range(1, self.num_observations):
            for j in range(self.num_states):
                alpha[t, j] = self.Emission[j, self.Observations[t]] * np.sum(alpha[t - 1] * self.Transition[:, j])
        return alpha

    def backward(self):
        """
        beta = ...
        """
        beta = np.zeros((self.num_observations, self.num_states))
        beta[-1] = 1  # Initialize last step
        for t in range(self.num_observations - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.sum(self.Transition[i, :] * self.Emission[:, self.Observations[t + 1]] * beta[t + 1, :])
        return beta

    def gamma_comp(self, alpha, beta):
        gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)
        return gamma

    def xi_comp(self, alpha, beta, gamma):
        xi = np.zeros((self.num_observations - 1, self.num_states, self.num_states))
        for t in range(self.num_observations - 1):
            denom = np.sum(alpha[t] * (self.Transition @ (self.Emission[:, self.Observations[t + 1]] * beta[t + 1])))
            xi[t] = (alpha[t][:, None] * self.Transition * self.Emission[:, self.Observations[t + 1]] * beta[
                t + 1]) / denom
        return xi

    def update(self, alpha, beta, gamma, xi):
        new_init_state = gamma[0]
        T_prime = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True).T
        M_prime = np.zeros_like(self.Emission)
        for j in range(self.num_states):
            for k in range(self.Emission.shape[1]):
                M_prime[j, k] = np.sum(gamma[:, j] * (self.Observations == k)) / np.sum(gamma[:, j])
        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):
        P_original = np.sum(alpha[-1])
        alpha_new = np.zeros_like(alpha)
        alpha_new[0] = new_init_state * M_prime[:, self.Observations[0]]
        for t in range(1, self.num_observations):
            for j in range(self.num_states):
                alpha_new[t, j] = M_prime[j, self.Observations[t]] * np.sum(alpha_new[t - 1] * T_prime[:, j])
        P_prime = np.sum(alpha_new[-1])
        return P_original, P_prime

""" UNCOMMENT THE BLOCK BELOW TO RUN RESULTS
    # MOST LIKELY STATE

    def most_likely_sequence(self, gamma):
        # Get the state with the highest probability at each time step
        most_likely_states = np.argmax(gamma, axis=1)
        # Convert state indices to city names
        most_likely_cities = [self.state_to_city[state] for state in most_likely_states]
        return most_likely_cities

    def display_gamma_table(self, gamma, most_likely_sequence):
        # Create a DataFrame from gamma
        gamma_df = pd.DataFrame(gamma, columns=['LA', 'NY'])

        # Add the corresponding city names (most likely sequence)
        gamma_df['Most Likely City'] = most_likely_sequence

        # Add time steps starting from 1
        gamma_df.index = gamma_df.index + 1
        gamma_df.index.name = 'Time Step'

        # Display the table
        print(gamma_df)

    def display_alpha_table(self, alpha):
        # Create a DataFrame from alpha
        alpha_df = pd.DataFrame(alpha, columns=['Alpha LA', 'Alpha NY'])

        # Add time steps starting from 1
        alpha_df.index = alpha_df.index + 1
        alpha_df.index.name = 'Time Step'

        # Display the table
        print(alpha_df)

    def display_beta_table(self, beta):
        # Create a DataFrame from beta
        beta_df = pd.DataFrame(beta, columns=['Beta LA', 'Beta NY'])

        # Add time steps starting from 1
        beta_df.index = beta_df.index + 1
        beta_df.index.name = 'Time Step'

        # Display the table
        print(beta_df)


if __name__ == '__main__':
    obs_matrix = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
    trans_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    obs_list = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
    init_dist = np.array((0.5, 0.5))
    hmm = HMM(obs_list, trans_matrix, obs_matrix, init_dist)

    alphas = hmm.forward()
    print("Alpha: \n", alphas)
    hmm.display_alpha_table(alphas)

    betas = hmm.backward()
    print("Beta: \n", betas)
    hmm.display_beta_table(betas)

    gammas = hmm.gamma_comp(alphas, betas)
    print("Gamma: \n", gammas)

    # Display gamma in table form
    print("The smoothing probabilities in table form, with most likely sequence")
    most_likely_sequence = hmm.most_likely_sequence(gammas)
    hmm.display_gamma_table(gammas, most_likely_sequence)

    print("Most likely sequence: \n", most_likely_sequence)

    xis = hmm.xi_comp(alphas, betas, gammas)
    print("Xi: \n", xis)

    T_prime, M_prime, new_init_state = hmm.update(alphas, betas, gammas, xis)
    print("Updated Transition Matrix: \n", T_prime)
    print("Updated Emission Matrix: \n", M_prime)
    print("Updated Initial State: \n", new_init_state)

    P_original, P_prime = hmm.trajectory_probability(alphas, betas, T_prime, M_prime, new_init_state)
    print("Original Observation Probability: ", P_original)
    print("Updated Observation Probability: ", P_prime)
    print("P_prime - P_original = ", P_prime - P_original)
"""
