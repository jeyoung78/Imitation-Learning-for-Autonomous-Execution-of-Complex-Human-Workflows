

import os
import argparse
import time
import numpy as np
import pybullet as p
import pybullet_data

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
    input_dim_w=10,     # ACT_DIM
    n_latent_dims=512,  # N_LATENT_DIMS
    vqvae_n_embed=32,   # VQVAE_N_EMBED
    vqvae_groups=2,     # VQVAE_GROUPS
    eval=True,
    device=DEVICE
)
vqvae.load_state_dict(torch.load("./checkpoints_pybullet/buttons/more_more_data/vqvae/vqvae_epoch200.pt", map_location=DEVICE))
# vqvae.eval()

# 2. Re-build GPT config and BehaviorTransformer
gpt_cfg = GPTConfig(
    input_dim=31,                # OBS_DIM
    output_dim=32 * 2,           # VQVAE_N_EMBED * VQVAE_GROUPS
    block_size=1,                # GPT_BLOCK_SIZE
    n_layer=12, n_head=8, n_embd=256,
    dropout=0.1
)
gpt = GPT(gpt_cfg).to(DEVICE)

agent = BehaviorTransformer(
    obs_dim=31, act_dim=10, goal_dim=0,
    gpt_model=gpt, vqvae_model=vqvae,
    offset_loss_multiplier=1e1,
    obs_window_size=1, act_window_size=1
).to(DEVICE)

mult = 2
# 3. Load your trained Transformer checkpoint0
checkpoint = torch.load("./checkpoints_pybullet/buttons/more_more_data/transformer/agent_epoch600.pt", map_location=DEVICE)
agent.load_state_dict(checkpoint)
agent.eval()

class ButtonPressEnv:
    """
    Environment: KUKA IIWA presses a sequence of moving buttons on a table.

    Enhanced with button characteristics (weight, size, color) that determine press order.

    - N buttons are placed and parameterized each episode with fixed or per-button weight/size/color.
    - Buttons are sorted ascending by size for pressing.
    - Press sequence for each button:
        1. Approach above the current button position.
        2. Lower to press.
        3. Retract above.

    Observations per step include:
      - Joint positions (M)
      - Joint velocities (M)
      - End-effector (EE) position (3)
      - All buttons: position (3), weight (1), size (1), color RGB (3), sorted index (1)
      - Step index within episode (1)
    Actions:
      - Target joint positions (M)
    """
    def __init__(self,
                 gui=False,
                 num_buttons=5,
                 approach_height=0.2,
                 press_height=0.05,
                 dt=1/180,
                 fixed_weight=1.0,
                 fixed_size=1.0,
                 fixed_color=None):
        self.gui = gui
        self.num_buttons = num_buttons
        self.approach_height = approach_height
        self.press_height = press_height
        self.dt = dt
        # fixed_weight/fixed_size: float or list length num_buttons
        self.fixed_weight = fixed_weight
        self.fixed_size = fixed_size
        # fixed_color: None, 3-length list, or list of per-button lists
        self.fixed_color = fixed_color

        mode = p.GUI if gui else p.DIRECT
        p.connect(mode)
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.3]
            )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        sdf = p.loadSDF(os.path.join(pybullet_data.getDataPath(),
                                     'kuka_iiwa', 'kuka_with_gripper2.sdf'))
        self.robot = sdf[0]

        # identify arm and finger joints
        self.arm_joints, self.finger_joints = [], []
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            name = info[1].decode().lower()
            if info[2] == p.JOINT_REVOLUTE:
                if 'finger' in name:
                    self.finger_joints.append(j)
                else:
                    self.arm_joints.append(j)
        self.ee_link = self.arm_joints[-1]

        # reset joint states
        init = [0,  0, 0.6, -1, 0, 1, 0, 0, 0,  1, 0, 0, 0, 0]
        for j in self.arm_joints + self.finger_joints:
            p.resetJointState(self.robot, j, 0)
            p.setJointMotorControl2(
                self.robot, j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=200
            )

        # spawn and parameterize buttons
        self.buttons = []
        for i in range(self.num_buttons):
            x = np.random.uniform(0.2, 1.0)
            y = np.random.uniform(-0.5, 0.5)
            z = 0.02
            weight = self.fixed_weight[i] if isinstance(self.fixed_weight, (list, tuple)) else self.fixed_weight
            size   = self.fixed_size[i]   if isinstance(self.fixed_size,   (list, tuple)) else self.fixed_size
            if self.fixed_color is None:
                color = np.random.rand(3).tolist()
            elif isinstance(self.fixed_color[0], (list, tuple)):
                color = list(self.fixed_color[i])
            else:
                color = list(self.fixed_color)

            uid = p.loadURDF('sphere_small.urdf', [x, y, z], globalScaling=size)
            p.changeDynamics(uid, -1, mass=weight)
            p.changeVisualShape(uid, -1, rgbaColor=color + [1.0])
            self.buttons.append({'uid': uid,
                                 'weight': weight,
                                 'size': size,
                                 'color': color})

        # sort by size
        self.buttons.sort(key=lambda b: b['size'])
        p.stepSimulation()

        # return full initial observation (step_idx=0)
        obs0 = self._get_obs(step_idx=0)
        return obs0

    def step(self, steps_per_phase=50):
        phases = ['approach', 'press', 'retract']
        all_obs, all_act, all_cc, all_cf = [], [], [], []
        idx = 0
        for sorted_idx, btn in enumerate(self.buttons):
            for phase in phases:
                duration = steps_per_phase if phase != 'press' else steps_per_phase // 2
                for s in range(duration):
                    bp = p.getBasePositionAndOrientation(btn['uid'])[0]
                    target_z = self.press_height if phase == 'press' else self.approach_height
                    target = [bp[0], bp[1], target_z]
                    ee = p.getLinkState(self.robot, self.ee_link)[4]
                    alpha = float(s + 1) / duration
                    waypoint = [ee[i] * (1 - alpha) + target[i] * alpha for i in range(3)]

                    jt = p.calculateInverseKinematics(self.robot, self.ee_link, waypoint)
                    action = list(jt[:len(self.arm_joints)])

                    for j_idx, j in enumerate(self.arm_joints):
                        p.setJointMotorControl2(
                            self.robot, j,
                            p.POSITION_CONTROL,
                            targetPosition=action[j_idx],
                            force=200)
                    for j in self.finger_joints:
                        p.setJointMotorControl2(
                            self.robot, j,
                            p.POSITION_CONTROL,
                            targetPosition=0.04,
                            force=50)

                    p.stepSimulation()
                    if self.gui:
                        time.sleep(self.dt)

                    obs = self._get_obs(step_idx=idx)
                    all_obs.append(obs)
                    all_act.append(action)
                    all_cc.append(0); all_cf.append(0.0)
                    idx += 1

        return (np.stack(all_obs, axis=0),
                np.stack(all_act, axis=0),
                np.array(all_cc, dtype=int),
                np.array(all_cf, dtype=float))

    def _get_obs(self, step_idx):
        # robot state
        jp, jv = [], []
        for j in self.arm_joints:
            pos, vel = p.getJointState(self.robot, j)[:2]
            jp.append(pos); jv.append(vel)
        ee_pos = p.getLinkState(self.robot, self.ee_link)[4]

        # collect all buttons
        btn_feats = []
        for sorted_idx, btn in enumerate(self.buttons):
            bp = p.getBasePositionAndOrientation(btn['uid'])[0]
            
            w, sz, col = btn['weight'], btn['size'], btn['color']
            btn_feats += [bp[0], bp[1], bp[2]]

            if step_idx <= 1:
                ball_idx = 0
            else:
                ball_idx = 1
        #vprint(btn_feats)
        # assemble full obs
        obs_vec = jp + jv + list(ee_pos) + btn_feats + [step_idx] + [ball_idx]
        return np.array(obs_vec, dtype=np.float32)
    
    def step_single(self, action, button, sorted_idx, step_idx):
        # apply one action and return obs, contacts
        # print(len(self.arm_joints))
        for j_idx,j in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot,j,p.POSITION_CONTROL,action[j_idx],force=200)
        for j in self.finger_joints:
            p.setJointMotorControl2(self.robot,j,p.POSITION_CONTROL,0.04,force=50)
        p.stepSimulation();
        if self.gui: 
            time.sleep(self.dt)
        obs = self._get_obs(step_idx)
        
        return obs, 0, 0.0


    def close(self):
        p.disconnect()

# ---------------- Evaluation Script ----------------
if __name__=='__main__':
    weights = [2.0, 3.0]       
    sizes   = [2.0, 3.0]       
    colors  = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
#       [0.0, 0.0, 1.0],
    ]
    parser = argparse.ArgumentParser(description='Evaluate trained policy')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--num_buttons', type=int, default=2)
    parser.add_argument('--steps_per_phase', type=int, default=int(50 * mult))
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()
    
    args.gui = True
    # Initialize environment
    env = ButtonPressEnv(gui=args.gui, num_buttons=args.num_buttons, fixed_weight=weights,
        fixed_size=sizes,
        fixed_color=colors)
    # Determine action dimension from env
    env.reset()
    action_dim = len(env.arm_joints)

    # Example placeholder policy
    def get_action(obs):
        obs_window = torch.from_numpy(obs).to(DEVICE)
        obs_window = obs_window.unsqueeze(0).unsqueeze(1)
        # print(obs_window.shape)
        # print(obs_window)
        with torch.no_grad():
            # The agent forward signature is (obs, goal, act); pass act=None for prediction
            predicted_continuous, loss, metrics = agent(obs_window, None, None)
            # predicted_continuous: shape [B * N_OBS_WINDOW, ACT_DIM] == [1, 14]
            next_action = predicted_continuous.squeeze(0).cpu().numpy()
            # print(next_action.shape)
        return next_action.reshape(-1)
    # Run evaluation episodes
    for ep in range(args.num_episodes):
        print(f"Eval Episode {ep+1}/{args.num_episodes}")
        obs = env.reset()
        # obs = np.append(obs, [0])
        obs = obs.astype(np.float32)        
        step_idx = 0
        # Press each button in sorted order
        for sorted_idx, btn in enumerate(env.buttons):
            for phase in ['approach', 'press', 'retract']:
                duration = args.steps_per_phase if phase != 'press' else args.steps_per_phase // 2
                for _ in range(duration):
                    # Compute action from policy
                    #print(obs[-1])
                    action = get_action(obs)
                    # print(action)
                    # Apply single step
                    obs, cc, cf = env.step_single(action, btn, sorted_idx, step_idx)

                    obs = obs.astype(np.float32)
                    
                    step_idx += (1/125)/mult
                    # print(obs)
                    print(step_idx)
    env.close()
