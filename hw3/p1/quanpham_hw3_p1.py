import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import math

# --- Constants ---
GRID_SIZE = 10
NUM_STATES = GRID_SIZE * GRID_SIZE
ACTIONS = ['N', 'E', 'S', 'W']
ACTION_PROBS = {'main': 0.7, 'side': 0.1, 'stay': 0.1}
DISCOUNT = 0.9

# --- Helper functions ---
def rc_to_idx(r, c):
    return r * GRID_SIZE + c

def idx_to_rc(idx):
    return divmod(idx, GRID_SIZE)  # returns (row, col)

def flip(coord):
    return (coord[1], coord[0])

# --- Define Obstacles, Initial and Goal States ---
obstacle_coords_main = [
    [9, 9], [8, 9], [7, 9], [6, 9], [5, 9], [4, 9], [3, 9], [2, 9], [1, 9], [0, 9],
    [9, 8], [9, 7], [9, 6], [9, 5], [9, 4], [9, 3], [9, 2], [9, 1], [9, 0],
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
    [3, 2], [4, 2], [5, 2], [6, 2],
    [4, 4], [4, 5], [4, 6], [4, 7], [5, 7],
    [7, 4], [7, 5]
]
# Remove duplicate obstacles
obstacle_coords_main = [list(x) for x in {tuple(coord) for coord in obstacle_coords_main}]
initial_main = (3, 6)
goal_main = (8, 1)

obstacles_rc = [flip(coord) for coord in obstacle_coords_main]
initial_rc = flip(initial_main)
goal_rc = flip(goal_main)

obstacle_indices = set(rc_to_idx(r, c) for (r, c) in obstacles_rc)
goal_idx = rc_to_idx(*goal_rc)
initial_idx = rc_to_idx(*initial_rc)

# --- Plot Functions ---
def plot_environment_ax(ax):
    # Draw grid cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor='white', edgecolor='black')
            ax.add_patch(rect)
    # Obstacles
    for s in obstacle_indices:
        r, c = idx_to_rc(s)
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                         facecolor='dimgray', edgecolor='black', alpha=1.0, zorder=10)
        ax.add_patch(rect)
    # Initial state in blue
    r, c = idx_to_rc(initial_idx)
    rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                     facecolor='blue', edgecolor='black', alpha=0.8, zorder=11)
    ax.add_patch(rect)
    # Goal state in green
    r, c = idx_to_rc(goal_idx)
    rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                     facecolor='green', edgecolor='black', alpha=0.8, zorder=11)
    ax.add_patch(rect)
    ax.set_title("Environment")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)

def plot_value_policy_ax(ax, J, policy, title):
    cmap = plt.cm.Reds
    vmin = np.min(J)
    vmax = np.percentile(J, 90)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Plot heatmap cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = rc_to_idx(r, c)
            color_val = cmap(norm(J[s]))
            rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor=color_val, edgecolor='black')
            ax.add_patch(rect)
            ax.text(c, r, f"{J[s]:.1f}", va='center', ha='center', color='black', fontsize=8)
    # Obstacles overlay
    for s in obstacle_indices:
        rr, cc = idx_to_rc(s)
        rect = Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                         facecolor='dimgray', edgecolor='black', alpha=1.0, zorder=10)
        ax.add_patch(rect)
    # Policy arrows for non-terminal states
    action_arrows = {'N': (0, 0.4), 'E': (0.4, 0), 'S': (0, -0.4), 'W': (-0.4, 0)}
    for s in range(NUM_STATES):
        if s in obstacle_indices or s == goal_idx or policy[s] == -1:
            continue
        rr, cc = idx_to_rc(s)
        act = ACTIONS[policy[s]]
        dx, dy = action_arrows[act]
        ax.arrow(cc, rr, dx, dy, head_width=0.15, head_length=0.15, fc='green', ec='green')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (col)", fontsize=8)
    ax.set_ylabel("y (row)", fontsize=8)
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.set_aspect('equal', 'box')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)

# --- Build Transition Matrix T and Cost Array q ---
T = np.zeros((NUM_STATES, NUM_STATES, len(ACTIONS)))
q = np.zeros((NUM_STATES, len(ACTIONS), NUM_STATES))

for s in range(NUM_STATES):
    if s in obstacle_indices:
        T[s, s, :] = 1.0
        q[s, :, s] = 10.0
    elif s == goal_idx:
        T[s, s, :] = 1.0
        q[s, :, s] = -10.0
    else:
        r, c = idx_to_rc(s)
        for a_idx, action in enumerate(ACTIONS):
            if action == 'N':
                intended = (r + 1, c)
                side1 = (r, c + 1)
                side2 = (r, c - 1)
            elif action == 'E':
                intended = (r, c + 1)
                side1 = (r + 1, c)
                side2 = (r - 1, c)
            elif action == 'S':
                intended = (r - 1, c)
                side1 = (r, c - 1)
                side2 = (r, c + 1)
            elif action == 'W':
                intended = (r, c - 1)
                side1 = (r - 1, c)
                side2 = (r + 1, c)
            def valid(pos):
                rr, cc = pos
                return 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE
            s_intended = rc_to_idx(*intended) if valid(intended) else s
            s_side1 = rc_to_idx(*side1) if valid(side1) else s
            s_side2 = rc_to_idx(*side2) if valid(side2) else s
            T[s, s_intended, a_idx] += ACTION_PROBS['main']
            T[s, s_side1, a_idx] += ACTION_PROBS['side']
            T[s, s_side2, a_idx] += ACTION_PROBS['side']
            T[s, s, a_idx] += ACTION_PROBS['stay']
            for s_prime in range(NUM_STATES):
                if T[s, s_prime, a_idx] > 0:
                    if s_prime in obstacle_indices:
                        q[s, a_idx, s_prime] = 10.0
                    elif s_prime == goal_idx:
                        q[s, a_idx, s_prime] = -10.0
                    else:
                        q[s, a_idx, s_prime] = 1.0

if not np.allclose(np.sum(T, axis=1), 1):
    raise ValueError("Transition matrix T rows do not sum to 1.")

# --- Policy Evaluation and Policy Iteration ---
east_idx = ACTIONS.index('E')
policy = np.full(NUM_STATES, east_idx, dtype=int)
for s in range(NUM_STATES):
    if s in obstacle_indices or s == goal_idx:
        policy[s] = -1

I = np.eye(NUM_STATES)

def evaluate_policy(policy):
    P_pi = np.zeros((NUM_STATES, NUM_STATES))
    r_pi = np.zeros(NUM_STATES)
    for s in range(NUM_STATES):
        if policy[s] == -1:
            P_pi[s, s] = 1.0
            r_pi[s] = 0.0
        else:
            a = policy[s]
            P_pi[s, :] = T[s, :, a]
            r_pi[s] = np.sum(T[s, :, a] * q[s, a, :])
    J = np.linalg.solve(I - DISCOUNT * P_pi, r_pi)
    return J

J_east = evaluate_policy(policy)

def policy_iteration(policy, max_iter=10):
    policy_history = []
    for it in range(max_iter):
        J_pi = evaluate_policy(policy)
        new_policy = np.copy(policy)
        for s in range(NUM_STATES):
            if s in obstacle_indices or s == goal_idx or policy[s] == -1:
                continue
            Q_sa = np.zeros(len(ACTIONS))
            for a in range(len(ACTIONS)):
                Q_sa[a] = np.sum(T[s, :, a] * (q[s, a, :] + DISCOUNT * J_pi))
            new_policy[s] = np.argmin(Q_sa)
        policy_history.append((J_pi.copy(), new_policy.copy()))
        if np.array_equal(policy, new_policy):
            print(f"Policy converged at iteration {it + 1}.")
            break
        policy = new_policy.copy()
    return policy, policy_history

final_policy, history = policy_iteration(policy, max_iter=10)
J_final = evaluate_policy(final_policy)

# --- List of Plot Data ---
plots = []
plots.append(('Environment', 'env', None))
plots.append(('Always-East Policy Evaluation (Initial)', 'value', (J_east, policy)))
for i, (J_iter, policy_iter) in enumerate(history, start=1):
    plots.append((f"Policy Iteration - Iteration {i}", 'value', (J_iter, policy_iter)))
plots.append(('Final Value Function and Policy', 'value', (J_final, final_policy)))

# --- Save Each Plot Individually ---
# Adjust the figsize and DPI for higher resolution if needed
for i, (title, plot_type, data) in enumerate(plots):
    fig, ax = plt.subplots(figsize=(6, 6))
    if plot_type == 'env':
        plot_environment_ax(ax)
    elif plot_type == 'value':
        J, pol = data
        plot_value_policy_ax(ax, J, pol, title)
    else:
        ax.set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    # Save each figure individually, here dpi and bbox_inches can improve output quality
    fig.savefig(f"plot_{i+1}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to free memory

print("Individual plots have been saved.")

# --- Create a Composite Figure with Subplots ---
n_plots = len(plots)
n_cols = math.ceil(math.sqrt(n_plots))
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.array(axes).flatten()

for ax, (title, plot_type, data) in zip(axes, plots):
    if plot_type == 'env':
        plot_environment_ax(ax)
    elif plot_type == 'value':
        J, pol = data
        plot_value_policy_ax(ax, J, pol, title)
    else:
        ax.set_visible(False)

# Hide extra axes if any.
for ax in axes[len(plots):]:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig("combined_subplots.png", dpi=300, bbox_inches='tight')
plt.show()








