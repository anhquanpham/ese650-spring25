import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''
        sensor_accuracy = 0.9
        sensor_error = 1 - sensor_accuracy
        move_success_prob = 0.9
        move_fail_prob = 1 - move_success_prob
        rows, cols = cmap.shape[1], cmap.shape[0]

        def apply_motion_model(prior_belief, movement):
            updated_belief = np.zeros([rows, cols])
            for x in range(rows):
                for y in range(cols):
                    if movement[1] == 0:  # Moving horizontally
                        if y + movement[0] < 0 or y + movement[0] > cmap.shape[1] - 1:
                            updated_belief[x][y] += prior_belief[x][y]  # Stay in place if out of bounds
                        else:
                            updated_belief[x][y] += move_fail_prob * prior_belief[x][y]
                            updated_belief[x][y + movement[0]] += move_success_prob * prior_belief[x][y]

                    if movement[0] == 0:  # Moving vertically
                        if x - movement[1] < 0 or x - movement[1] > cmap.shape[0] - 1:
                            updated_belief[x][y] += prior_belief[x][y]  # Stay in place if out of bounds
                        else:
                            updated_belief[x][y] += move_fail_prob * prior_belief[x][y]
                            updated_belief[x - movement[1]][y] += move_success_prob * prior_belief[x][y]
            return updated_belief


        def apply_sensor_model(predicted_belief, observed_color):
            corrected_belief = np.zeros([rows, cols])
            normalization_factor = 0

            for x in range(rows):
                for y in range(cols):
                    correct_observation = int(cmap[x][y] == observed_color)
                    corrected_belief[x][y] = predicted_belief[x][y] * (
                            correct_observation * sensor_accuracy + (1 - correct_observation) * sensor_error)
                    normalization_factor += corrected_belief[x][y]
            corrected_belief /= normalization_factor
            return corrected_belief

        belief = apply_motion_model(belief, action)
        belief = apply_sensor_model(belief, observation)
        return belief


