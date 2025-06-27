####BEST VERSION SO FAR

import os
os.environ["MUJOCO_GL"] = "egl"


from dm_control import suite,viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt


# right after your imports:
CHECKPOINT_DIR = "checkpoints_v2_powerful"
PLOT_DIR       = "plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Existing Helper and Policy Skeleton ===
# Helper to convert observation dict to flat state
def obs_to_state(obs):
    return np.array(obs['orientations'].tolist() + [obs['height']] + obs['velocity'].tolist(), dtype=np.float32)

class uth_t(nn.Module):
    def __init__(s, xdim, udim, hdim=512, fixed_var=True):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.fixed_var = fixed_var
        s.fc1     = nn.Linear(xdim, hdim)
        s.fc2     = nn.Linear(hdim, hdim)
        s.mu_head = nn.Linear(hdim, udim)
        if fixed_var:
            s.log_std = nn.Parameter(torch.zeros(udim))
        else:
            s.std_head = nn.Linear(hdim, udim)

    def forward(s, x):
        h = F.relu(s.fc1(x))
        h = F.relu(s.fc2(h))
        mu = s.mu_head(h)
        if s.fixed_var:
            std = torch.exp(s.log_std.to(x.device))
        else:
            std = torch.exp(s.std_head(h))
        return mu, std

# Provided rollout
def rollout(e, uth, T=1000):
    traj = []
    t = e.reset()
    x = obs_to_state(t.observation)
    for _ in range(T):
        with torch.no_grad():
            u, _ = uth(torch.from_numpy(x).float().unsqueeze(0).to(device))
        r = e.step(u.cpu().numpy())
        x_next = obs_to_state(r.observation)
        traj.append({'xp': x_next, 'r': r.reward, 'u': u.cpu().numpy(), 'd': r.last()})
        x = x_next
        if r.last():
            break
    return traj

# === Original Usage and GIF Replay ===
r0 = np.random.RandomState(37)
e = suite.load('walker', 'walk', task_kwargs={'random': r0})
U = e.action_spec(); udim = U.shape[0]
xdim = obs_to_state(e.reset().observation).shape[0]

uth = uth_t(xdim, udim).to(device)
def nn_policy(time_step):
    x = obs_to_state(time_step.observation)
    with torch.no_grad():
        mu, std = uth(torch.from_numpy(x).unsqueeze(0).to(device))
        return mu.squeeze(0).cpu().numpy()
#viewer.launch(e, policy=nn_policy)

traj = rollout(e, uth)
print("Trajectory length:", len(traj))
e2 = suite.load('walker', 'walk', task_kwargs={'random': r0})
time_step = e2.reset()
frames = [e2.physics.render(height=480, width=640, camera_id=0)]
for u in [step['u'] for step in traj]:
    time_step = e2.step(u)
    frames.append(e2.physics.render(height=480, width=640, camera_id=0))
imageio.mimsave('trajectory_v2_powerful_initial.gif', frames, fps=30)
print("Saved replay to trajectory_v2_powerful_initial.gif")

# === PPO Implementation ===


# Ensure reproducibility
torch.manual_seed(37)

# Value network (critic)
class Critic(nn.Module):
    def __init__(self, xdim, hdim=512):
        super().__init__()
        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.v_head = nn.Linear(hdim, 1)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.v_head(h).squeeze(-1)

# Collect one on-policy episode: compute GAE as before, but also return raw episode return
def collect_episode(env, actor, critic, max_steps, gamma=0.99, lam=0.95):
    actor.train(); critic.train()
    states, actions, logps, rewards, vals = [], [], [], [], []
    t = env.reset()
    s = obs_to_state(t.observation)
    done = False
    for _ in range(max_steps):
        states.append(s.copy())
        st = torch.from_numpy(np.array(s, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std = actor(st)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            lp = dist.log_prob(a).sum(-1)
        t = env.step(a.squeeze(0).cpu().numpy())
        r = t.reward
        rewards.append(r)
        actions.append(a.cpu().squeeze(0))
        logps.append(lp.cpu().squeeze(0))
        vals.append(critic(st).item())
        s = obs_to_state(t.observation)
        done = t.last()
        if done: break
    # raw episode return
    ep_ret = sum(rewards)
    # last value for bootstrapping
    st = torch.from_numpy(np.array(s, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad(): last_val = 0.0 if done else critic(st).item()
    vals.append(last_val)
    # GAE advantage
    advs, rets = [], []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * vals[i+1] - vals[i]
        gae   = delta + gamma * lam * gae
        advs.insert(0, gae)
        rets.insert(0, gae + vals[i])
    returns = torch.tensor(rets, dtype=torch.float32, device=device)
    advs    = torch.tensor(advs, dtype=torch.float32, device=device)
    advs    = (advs - advs.mean()) / (advs.std() + 1e-8)
    # fast batching
    states_t  = torch.from_numpy(np.stack(states).astype(np.float32)).to(device)
    actions_t = torch.stack(actions).to(device)
    logps_t   = torch.stack(logps).to(device)
    return (states_t, actions_t, logps_t, returns, advs), ep_ret


# def collect_episodes(env, actor, critic, max_steps_per_ep, n_episodes=8):
#     batched = []
#     ep_rets = []
#     for _ in range(n_episodes):
#         data, ep_ret = collect_episode(env, actor, critic, max_steps_per_ep)
#         batched.append(data)
#         ep_rets.append(ep_ret)

#     # Unpack and concatenate
#     states   = torch.cat([d[0] for d in batched], dim=0)
#     actions  = torch.cat([d[1] for d in batched], dim=0)
#     logps    = torch.cat([d[2] for d in batched], dim=0)
#     returns  = torch.cat([d[3] for d in batched], dim=0)
#     advs     = torch.cat([d[4] for d in batched], dim=0)

#     return (states, actions, logps, returns, advs), np.mean(ep_rets)


# PPO update step (unchanged)
def ppo_step(actor, critic, opt, states, actions, old_logps, returns, advs,
             clip=0.2, c1=0.5, c2=0.01, epochs=4):
    actor.train(); critic.train()
    for _ in range(epochs):
        mu, std = actor(states)
        dist = torch.distributions.Normal(mu, std)
        logps = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(logps - old_logps)
        s1 = ratio * advs
        s2 = torch.clamp(ratio, 1-clip, 1+clip) * advs
        aloss = -torch.min(s1, s2).mean()
        vpred = critic(states)
        closs = F.mse_loss(vpred, returns)
        ent   = dist.entropy().sum(-1).mean()
        loss  = aloss + c1 * closs - c2 * ent
        opt.zero_grad(); loss.backward(); opt.step()
    return aloss.item(), closs.item(), ent.item()

def evaluate_policy(env, actor, episodes=10):
    actor.eval()
    returns = []
    for _ in range(episodes):
        t = env.reset()
        s = obs_to_state(t.observation)
        done = False
        total_r = 0.0
        while not done:
            with torch.no_grad():
                mu, _ = actor(torch.from_numpy(s).float().unsqueeze(0).to(device))
                a = mu.squeeze(0).cpu().numpy()
            t = env.step(a)
            total_r += t.reward
            s = obs_to_state(t.observation)
            done = t.last()
        returns.append(total_r)
    return np.mean(returns)

# Train PPO: log raw episode returns
def train_ppo():
    seed = 37
    env  = suite.load('walker','walk', task_kwargs={'random': np.random.RandomState(seed)})
    xdim = obs_to_state(env.reset().observation).shape[0]
    udim = env.action_spec().shape[0]
    actor  = uth_t(xdim, udim, hdim=512).to(device)
    critic = Critic(xdim).to(device)
    opt     = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    timesteps = 0
    total_steps = 20_000_000
    log_steps, log_returns = [], []
    # while timesteps < total_steps:
    #     (states, actions, logps, returns, advs), ep_ret = collect_episode(env, actor, critic, 1000)
    #     batch_size = returns.size(0)
    #     timesteps += batch_size
    #     a_loss, c_loss, ent = ppo_step(actor, critic, opt, states, actions, logps, returns, advs)
    #     log_steps.append(timesteps)
    #     log_returns.append(ep_ret)
    #     if timesteps % 50000 < batch_size:
    #         print(f"Steps: {timesteps}, Episode Return: {ep_ret:.2f}", flush=True)
    eval_interval = 10000
    next_eval = eval_interval
    best_return = -float('inf')  # Initialize best return

    while timesteps < total_steps:
        (states, actions, logps, returns, advs), ep_ret = collect_episode(env, actor, critic, 1000)
        batch_size = returns.size(0)
        timesteps += batch_size

        # --- Linear learning rate decay ---
        progress = timesteps / total_steps  # goes from 0.0 → 1.0
        lr_now = 3e-4 * (1.0 - progress)
        for param_group in opt.param_groups:
            param_group['lr'] = lr_now
        
        a_loss, c_loss, ent = ppo_step(actor, critic, opt, states, actions, logps, returns, advs)

        # Evaluate every `eval_interval` timesteps
        if timesteps >= next_eval:
            avg_eval_return = evaluate_policy(env, actor, episodes=10)
            print(f"[Eval @ {timesteps} steps] Avg Return: {avg_eval_return:.2f}")
            log_steps.append(timesteps)
            log_returns.append(avg_eval_return)
            next_eval += eval_interval
            # Save model every evaluation interval
            if avg_eval_return > best_return:
                best_return = avg_eval_return
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'timesteps': timesteps,
                    'best_return': best_return
                }, os.path.join(CHECKPOINT_DIR, f'best_model_v2_powerful.pt'))
                print(f" Saved new best model with return: {best_return:.2f}")
    plt.plot(log_steps, log_returns)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Return')
    plt.title('PPO on Walker')
    plt.savefig(os.path.join(PLOT_DIR, "ppo_walker_returns_v2_powerful.png"))
    plt.close()

def load_best_and_render():
    print("Loading best checkpoint and rendering replay...")

    # Reload environment (same seed for deterministic rendering)
    env = suite.load('walker', 'walk', task_kwargs={'random': np.random.RandomState(37)})
    xdim = obs_to_state(env.reset().observation).shape[0]
    udim = env.action_spec().shape[0]

    # Create actor network (must match trained architecture)
    actor = uth_t(xdim, udim, hdim=512, fixed_var=True).to(device)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_v2_powerful.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # Rollout for rendering
    frames = []
    t = env.reset()
    s = obs_to_state(t.observation)
    frames.append(env.physics.render(height=480, width=640, camera_id=0))

    while not t.last():
        with torch.no_grad():
            x = torch.from_numpy(np.array(s, dtype=np.float32)).unsqueeze(0).to(device)
            mu, _ = actor(x)
            a = mu.squeeze(0).cpu().numpy()
        t = env.step(a)
        s = obs_to_state(t.observation)
        frames.append(env.physics.render(height=480, width=640, camera_id=0))

    imageio.mimsave("walker_best_v2_powerful.gif", frames, fps=30)
    print("Saved best replay to walker_best_v2_powerful.gif")

if __name__ == '__main__':
    print("hdim=512, steps = 20M, linear decay learning rate")
    train_ppo()
    load_best_and_render()
    
