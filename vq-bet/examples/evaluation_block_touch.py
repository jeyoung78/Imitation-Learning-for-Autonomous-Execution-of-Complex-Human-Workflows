
import sys, pathlib
# make the repo root importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from vqvae.vqvae import VqVae
from vq_behavior_transformer.gpt import GPT, GPTConfig
from vq_behavior_transformer.bet import BehaviorTransformer
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Re-build VQ-VAE (must match pretraining)
vqvae = VqVae(
    input_dim_h=1,      # ACTION_WINDOW_SIZE
    input_dim_w=14,     # ACT_DIM
    n_latent_dims=512,  # N_LATENT_DIMS
    vqvae_n_embed=16,   # VQVAE_N_EMBED
    vqvae_groups=2,     # VQVAE_GROUPS
    eval=True,
    device=DEVICE
)
vqvae.load_state_dict(torch.load("./checkpoints_pybullet/block_touch/vqvae/vqvae_epoch300.pt", map_location=DEVICE))
# vqvae.eval()

# 2. Re-build GPT config and BehaviorTransformer
gpt_cfg = GPTConfig(
    input_dim=36,                # OBS_DIM
    output_dim=16 * 2,           # VQVAE_N_EMBED * VQVAE_GROUPS
    block_size=1,                # GPT_BLOCK_SIZE
    n_layer=6, n_head=8, n_embd=128,
    dropout=0.1
)
gpt = GPT(gpt_cfg).to(DEVICE)

agent = BehaviorTransformer(
    obs_dim=36, act_dim=14, goal_dim=0,
    gpt_model=gpt, vqvae_model=vqvae,
    offset_loss_multiplier=1e3,
    obs_window_size=1, act_window_size=1
).to(DEVICE)

# 3. Load your trained Transformer checkpoint
checkpoint = torch.load("./checkpoints_pybullet/block_touch/transformer/agent_epoch2000.pt", map_location=DEVICE)
agent.load_state_dict(checkpoint)
agent.eval()

# 4. Inference: predict the next action for a single obs window
#    Suppose you have `obs_window` of shape (1, 35), e.g. numpy or torch tensor
# dummy = np.zeros(35, dtype=np.float32)
# obs_np = np.array([-0.002324990462511778, 0.005895666778087616, -0.0013503036461770535, -0.01807596907019615, -0.0009022774174809456, 0.014715025201439857, -0.0011282835621386766, -0.0009071915410459042, 0.04003555327653885, 0.04005879536271095, 0.00888172909617424, 0.039883870631456375, 0.04031004756689072, -0.00017535529332235456], dtype=np.float32)

# 2) to tensor [1, 1, 35]:
def get_acts(obs_np):
    obs_window = torch.from_numpy(obs_np).to(DEVICE)
    obs_window = obs_window.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        # The agent forward signature is (obs, goal, act); pass act=None for prediction
        predicted_continuous, loss, metrics = agent(obs_window, None, None)
        # predicted_continuous: shape [B * N_OBS_WINDOW, ACT_DIM] == [1, 14]
        next_action = predicted_continuous.squeeze(0).cpu().numpy()

    return next_action

import pybullet as p
import pybullet_data
import numpy as np
import time

def find_finger_joints(robot):
    return sorted([j for j in range(p.getNumJoints(robot))
                   if 'finger' in p.getJointInfo(robot, j)[1].decode().lower()])

class SimpleManipulationEnv:
    def __init__(self, gui=False):
        self.gui = gui
        if gui:
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setRealTimeSimulation(1)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.1]
            )
        else:
            p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        sdf = p.loadSDF(pybullet_data.getDataPath() + "/kuka_iiwa/kuka_with_gripper2.sdf")
        self.robot = sdf[0]
        self.num_joints = p.getNumJoints(self.robot)
        self.finger_joints = find_finger_joints(self.robot)
        self.current_action = [0.0] * self.num_joints
        y = np.random.uniform(-0.2, 0.2)
        self.cube = p.loadURDF(
            "cube_small.urdf",
            [0.3, -0.8, 0.1],
            p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=1.8
        )
        for j in self.finger_joints:
            self.current_action[j] = 0.04
        return self._get_obs()

    def step(self, action):
        for j, target in enumerate(action):
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200
            )
        p.stepSimulation()
        if self.gui:
            time.sleep(1.0 / 240.0)
        self.current_action = action.copy()
        return self._get_obs()

    def _get_obs(self):
        js = p.getJointStates(self.robot, list(range(self.num_joints)))
        pos = [s[0] for s in js]
        vel = [s[1] for s in js]
        cube_p, cube_o = p.getBasePositionAndOrientation(self.cube)
        return np.array(pos + vel + list(cube_p) + list(cube_o), dtype=np.float32)

    def close(self):
        p.disconnect()

def run_fixed_action(steps=500, gui=True):
    env = SimpleManipulationEnv(gui=gui)
    obs = env.reset()
    obs = np.append(obs, [0])
    obs = obs.astype(np.float32)  

    action_14d = get_acts(obs)
    action = np.asarray(action_14d).ravel().tolist()

    # 3) loop
    for i in range(steps):
        obs = env.step(action)
        obs = np.append(obs, i)
        obs = obs.astype(np.float32)  
        action_14d = get_acts(obs)
        action = np.asarray(action_14d).ravel().tolist()
        print(i)
        # if GUI mode, slow it down to real time
        if gui:
            time.sleep(1.0/120.0)

    env.close()


if __name__ == "__main__":
    # Example: all-zeros action
    run_fixed_action()
