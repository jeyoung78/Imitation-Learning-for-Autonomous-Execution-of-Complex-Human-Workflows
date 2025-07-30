"""
import pybullet as p
import pybullet_data
import argparse
import numpy as np
import os
import time

# Utility to find gripper finger joints by name
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
        open_val = 0.04
        # Initialize gripper open
        for j in self.finger_joints:
            # instantaneously set the joint position
            p.resetJointState(self.robot, j, open_val)
            # ensure the motor keeps it open
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=open_val,
                force=100
            )
        self.current_action = [0.0] * self.num_joints
        y = np.random.uniform(-0.8, 0.8)
        x = np.random.uniform(0.4, 0.8)
        self.cube = p.loadURDF(
            "cube_small.urdf",
            [x, y, 0.1],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        for j in self.finger_joints:
            self.current_action[j] = 0.04
        return self._get_obs(), x, y

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


def collect_contact_demo(args):
    env = SimpleManipulationEnv(gui=args.gui)

    # Top‐level storage for all episodes
    all_obs = []
    all_act = []
    all_contact_counts = []
    all_contact_forces = []

    for ep in range(args.num_episodes):
        obs_list = []
        act_list = []
        contact_counts = []
        contact_forces = []

        # Reset environment & record initial state
        obs, x, y = env.reset()
        idx = 0
        obs = np.append(obs, idx)
        act = env.current_action.copy()
        obs_list.append(obs)
        act_list.append(act)

        # Compute grasp waypoints
        cube_p, _ = p.getBasePositionAndOrientation(env.cube)
        pre_grasp = [cube_p[0], cube_p[1], cube_p[2] + 0.2]
        grasp     = [cube_p[0], cube_p[1], cube_p[2] + 0.02]
        joints_pre   = list(p.calculateInverseKinematics(env.robot, env.num_joints-1, pre_grasp))
        joints_grasp = list(p.calculateInverseKinematics(env.robot, env.num_joints-1, grasp))

        # Open/close target values
        open_val, close_val = 0.04, 0.0
        phases = [
            (joints_pre,   open_val,  100),
            (joints_grasp, open_val,  100),
            (joints_grasp, close_val, 100),
            (joints_pre,   close_val, 200),
        ]

        current = env.current_action.copy()
        
        for tgt_joints, finger_tgt, steps in phases:
            # build full target vector
            full_tgt = tgt_joints + [current[j] for j in range(len(tgt_joints), env.num_joints)]
            for j in env.finger_joints:
                full_tgt[j] = finger_tgt

            for step in range(steps):
                α = (step + 1) / steps
                action = [(1 - α) * c + α * t for c, t in zip(current, full_tgt)]
                obs = env.step(action)
                obs = np.append(obs, idx)
                idx = idx + 1
                # print(obs)
                obs_list.append(obs)
                act_list.append(action.copy())

                # log contacts
                contacts = p.getContactPoints(env.robot, env.cube)
                contact_counts.append(len(contacts))
                contact_forces.append(sum(pt[9] for pt in contacts))

            current = full_tgt.copy()

        # Episode complete: aggregate
        all_obs.append(np.stack(obs_list, axis=0))
        all_act.append(np.stack(act_list, axis=0))
        all_contact_counts.append(np.array(contact_counts, dtype=int))
        all_contact_forces.append(np.array(contact_forces, dtype=float))
        # print(obs)

    # Persist to disk
    
    
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "data_obs.npy"),
            np.stack(all_obs, axis=0))
    np.save(os.path.join(args.save_dir, "data_act.npy"),
            np.stack(all_act, axis=0))
    np.save(os.path.join(args.save_dir, "data_contact_counts.npy"),
            np.stack(all_contact_counts, axis=0))
    np.save(os.path.join(args.save_dir, "data_contact_forces.npy"),
            np.stack(all_contact_forces, axis=0))
    
    print(f"Saved {len(all_obs)} episodes of contact‐based demos to '{args.save_dir}'")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect contact-based pick-up demo for VQ-BET")
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=400)
    args = parser.parse_args()
    collect_contact_demo(args)
"""
import os
import argparse
import time
import numpy as np
import pybullet as p
import pybullet_data


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
                 dt=1/240.,
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
        for j in self.arm_joints + self.finger_joints:
            p.resetJointState(self.robot, j, 0.0)
            p.setJointMotorControl2(
                self.robot, j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=200
            )

        # spawn and parameterize buttons
        self.buttons = []
        for i in range(self.num_buttons):
            x = np.random.uniform(0.4, 0.8)
            y = np.random.uniform(-0.4, 0.4)
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
                    #print(len(action))
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
                    idx += 1/125

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

        # assemble full obs
        if step_idx <= 1:
            ball_idx = 0
        else:
            ball_idx = 1

        obs_vec = jp + jv + list(ee_pos) + btn_feats + [step_idx] + [ball_idx]
        return np.array(obs_vec, dtype=np.float32)
    

    def close(self):
        p.disconnect()

def collect_buttonpress_data(args):
    # --- per‐button parameters must all have length == args.num_buttons
    weights = [1.0, 2.0]       # len=3
    sizes   = [1.0, 2.0]       # len=3
    colors  = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
#        [0.0, 0.0, 1.0],
    ]
    assert args.num_buttons == len(weights) == len(sizes) == len(colors), \
           "num_buttons must equal len(weights/sizes/colors)"

    env = ButtonPressEnv(
        gui=args.gui,
        num_buttons=args.num_buttons,
        approach_height=args.approach_height,
        press_height=args.press_height,
        dt=args.dt,
        fixed_weight=weights,
        fixed_size=sizes,
        fixed_color=colors
    )

    all_obs, all_act, all_cc, all_cf = [], [], [], []
    for ep in range(args.num_episodes):
        print(f"Episode {ep+1}/{args.num_episodes}")

        # 1) reset → returns obs_0 containing all buttons
        obs0 = env.reset()

        

        # 2) collect the rest of the trajectory
        obs_seq, act_seq, cc_seq, cf_seq = env.step(steps_per_phase=args.steps_per_phase)

        #if obs0.shape[0] == obs_seq.shape[1] - 1:
        #   obs0 = np.concatenate([obs0, [0]], axis=0)
            

        # 3) prepend initial obs if you want a continuous trace
        # obs_full = np.vstack([obs0[np.newaxis], obs_seq])

        all_obs.append(obs_seq)
        all_act.append(act_seq)
        all_cc.append(cc_seq)
        all_cf.append(cf_seq)
        # print(obs_full[-1])

    env.close()

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'data_obs.npy'), np.stack(all_obs, axis=0))
    np.save(os.path.join(args.save_dir, 'data_act.npy'), np.stack(all_act, axis=0))
    np.save(os.path.join(args.save_dir, 'data_contact_counts.npy'),
            np.stack(all_cc, axis=0))
    np.save(os.path.join(args.save_dir, 'data_contact_forces.npy'),
            np.stack(all_cf, axis=0))

    print(f"Saved {len(all_obs)} button-press episodes to '{args.save_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect weighted button-press data')
    parser.add_argument('--save_dir', type=str, default='data/button')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--dt', type=float, default=1/240.)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--num_buttons', type=int, default=2)
    parser.add_argument('--steps_per_phase', type=int, default=50)
    parser.add_argument('--approach_height', type=float, default=0.5)
    parser.add_argument('--press_height', type=float, default=0.05)
    args = parser.parse_args()
    collect_buttonpress_data(args)