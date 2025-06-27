import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    print("Correct belief_states: \n", belief_states)
    print("Correct belief_states shape: \n", belief_states.shape)

    #### Test your code here
    hf = HistogramFilter()

    # Initialize uniform belief over entire grid
    belief = np.ones_like(cmap, dtype=np.float32) / cmap.size

    estimated_positions = []

    for t in range(len(actions)):
        belief = hf.histogram_filter(cmap, belief, actions[t], observations[t])
        best_guess = np.argwhere(belief.max() == belief)[0]
        best_guess[0], best_guess[1] = best_guess[1], belief.shape[0] - 1 - best_guess[0]
        estimated_positions.append(best_guess)  # Store (row, col)

        #print("Updated Belief Matrix:", belief.shape)
        #print("Estimated Position:", best_guess)

    estimated_positions = np.array(estimated_positions)
    print("Calculated belief states: \n", estimated_positions)
    print("Calculated belief states shape: \n",estimated_positions.shape)

