import sys, pathlib
# make the repo root importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
import torch
import numpy as np
from pathlib import Path
from vqvae.vqvae import VqVae
from vq_behavior_transformer.gpt import GPT, GPTConfig
from vq_behavior_transformer.bet import BehaviorTransformer
import gym             # or your UR3‐specific env package
# from gym_ur3 import UR3Env  # if you have a custom UR3 wrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Paths ─────
VQVAE_CKPT    = "./checkpoints/ur3_vqvae/vqvae_epoch150.pt"
TRANSFORMER_CKPT = r"C:\Users\jupar\imitationLearning\vq-bet\checkpoints\ur3_transformer\agent_epoch080.pt"
NUM_EPISODES  = 20
MAX_STEPS     = 200     # or whatever your UR3 task horizon is

# ───── Rebuild models ─────
# 1) Load pretrained VQ-VAE
vqvae = VqVae(
    input_dim_h=1,
    input_dim_w=2,
    n_latent_dims=512,
    vqvae_n_embed=16,
    vqvae_groups=2,
    eval=True,
    device=DEVICE    # <— this already moves internal modules
)
vqvae.load_state_dict(torch.load(VQVAE_CKPT, map_location=DEVICE))

# 2) Build GPT
gpt_cfg = GPTConfig(
    input_dim=6,                 # OBS_DIM your dataset used
    output_dim=16 * 2,           # vqvae_n_embed * vqvae_groups
    block_size=1,                # N_OBS_WINDOW
    n_layer=6,
    n_head=8,
    n_embd=128,
    dropout=0.1
)
gpt = GPT(gpt_cfg).to(DEVICE)

# 3) Build BehaviorTransformer and load your trained weights
agent = BehaviorTransformer(
    obs_dim=6,
    act_dim=2,
    goal_dim=0,
    gpt_model=gpt,
    vqvae_model=vqvae,
    offset_loss_multiplier=1e3,
    obs_window_size=1,
    act_window_size=1
).to(DEVICE)

ckpt = torch.load(TRANSFORMER_CKPT, map_location=DEVICE)
agent.load_state_dict(ckpt)
agent.eval()

# ───── Create environment ─────
# Replace "UR3-v0" with your actual gym ID or custom env constructor.
# env = gym.make("UR3-v0")  
# If you have a custom class:
# env = UR3Env(...)

# Optionally wrap for video:
# from gym.wrappers import Monitor
# env = Monitor(env, "./eval_videos", force=True)
#from envs.ur3.gym_custom.envs.custom import UR3PracticeEnv
from gym_custom.envs.custom.ur3_env import UR3PracticeEnv
env = UR3PracticeEnv()

# ───── Evaluation loop ─────
all_returns = []
for ep in range(NUM_EPISODES):
    obs = env.reset()
    obs_hist = [obs]    # for obs_window_size=1, we only need last obs
    episode_return = 0.0

    for t in range(MAX_STEPS):
        # 1) Prepare observation tensor
        obs_tensor = (
            torch.from_numpy(np.array(obs_hist[-1])[None, None, :])
            .float().to(DEVICE)
        )  # shape: (1, 1, OBS_DIM)

        # 2) Run policy to get discrete codes → continuous actions
        with torch.no_grad():
            # agent returns: predicted_actions, loss, metrics
            pred_actions, _, _ = agent(obs_tensor, None, None)
            # shape (batch⋅window, ACT_DIM) = (1⋅1, 2)
            action = pred_actions[0].cpu().numpy()

        # 3) Step in the sim
        obs, reward, done, info = env.step(action)
        episode_return += reward
        obs_hist.append(obs)

        if done:
            break

    all_returns.append(episode_return)
    print(f"Episode {ep+1:02d} return: {episode_return:.3f}")

# ───── Summarize ─────
avg_ret = np.mean(all_returns)
success_rate = np.mean([ret >= YOUR_SUCCESS_THRESHOLD for ret in all_returns])
print(f"\nAverage Return over {NUM_EPISODES} episodes: {avg_ret:.3f}")
print(f"Success Rate: {success_rate*100:.1f}%")
