
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

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# Utility: find finger joints of the gripper
def find_finger_joints(robot_id):
    return [j for j in range(p.getNumJoints(robot_id))
            if 'finger' in p.getJointInfo(robot_id, j)[1].decode().lower()]

class GraspRecorder:
    OPEN_POS = 0.10       # gripper open position
    CLOSE_POS = 0.0       # gripper closed
    OBJECT_SCALE = 1.8    # cube scale
    STEP_DELAY = 1/240.

    def __init__(self, save_dir='data', num_episodes=10, gui=True):
        self.save_dir = save_dir
        self.num_episodes = num_episodes
        self.gui = gui
        self._init_pybullet()

    def _init_pybullet(self):
        if self.gui:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.2, 45, -30, [0.5, 0, 0.2])
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def _load_robot(self):
        p.loadURDF("plane.urdf")
        self.robot = p.loadSDF(pybullet_data.getDataPath() + "/kuka_iiwa/kuka_with_gripper2.sdf")[0]
        self.num_joints = p.getNumJoints(self.robot)
        self.finger_joints = find_finger_joints(self.robot)
        for j in range(self.num_joints):
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)

    def control_gripper(self, pos, force=250):
        for j in self.finger_joints:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                     targetPosition=pos, force=force)

    def reset_episode(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._load_robot()
        # randomize cube pose
        x = np.random.uniform(0.4, 0.6)
        y = np.random.uniform(-0.2, 0.2)
        yaw = np.random.uniform(0, 2 * np.pi)
        pos = [x, y, 0.1]
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        self.cube = p.loadURDF(
            "cube_small.urdf", pos, quat,
            globalScaling=self.OBJECT_SCALE
        )
        # initial open
        self.control_gripper(self.OPEN_POS)
        # initialize storage
        obs_list, act_list = [], []
        contact_counts, contact_forces = [], []
        # record initial obs/action
        obs_list.append(self.get_obs())
        act_list.append([0.0] * self.num_joints)
        return obs_list, act_list, contact_counts, contact_forces, pos

    def get_obs(self):
        # joint states
        js = p.getJointStates(self.robot, list(range(self.num_joints)))
        pos = [s[0] for s in js]
        vel = [s[1] for s in js]
        # cube pose
        cube_p, cube_o = p.getBasePositionAndOrientation(self.cube)
        return np.array(pos + vel + list(cube_p) + list(cube_o), dtype=np.float32)

    def step(self, action):
        # apply arm + gripper actions
        for j, tgt in enumerate(action):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                     targetPosition=tgt, force=200)
        p.stepSimulation()
        if self.gui:
            time.sleep(self.STEP_DELAY)
        return self.get_obs()

    def attach_constraint(self):
        wrist = self.num_joints - 1
        p.createConstraint(self.robot, wrist,
                           self.cube, -1,
                           p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

    def run(self):
        all_obs, all_act = [], []
        all_counts, all_forces = [], []
        for ep in range(self.num_episodes):
            obs_list, act_list, counts, forces, cube_pos = self.reset_episode()
            # phases: approach, descend, close, lift
            phases = []
            # 1) approach
            above = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.3]
            j_above = p.calculateInverseKinematics(self.robot, self.num_joints-1, above)
            phases.append((j_above, self.OPEN_POS, 100))
            # 2) descend
            below = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.05]
            j_below = p.calculateInverseKinematics(self.robot, self.num_joints-1, below)
            phases.append((j_below, self.OPEN_POS, 120))
            # 3) close & attach
            phases.append((j_below, self.CLOSE_POS, 60, True))
            # 4) lift
            lift = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.5]
            j_lift = p.calculateInverseKinematics(self.robot, self.num_joints-1, lift)
            phases.append((j_lift, self.CLOSE_POS, 140))

            for phase in phases:
                joints, grip, steps = phase[0], phase[1], phase[2]
                attach = len(phase) == 4 and phase[3]
                for _ in range(steps):
                    # build full action vector
                    action = list(joints) + [0.0] * (self.num_joints - len(joints))
                    # override finger joints
                    for j in self.finger_joints:
                        action[j] = grip
                    # step
                    obs = self.step(action)
                    act_list.append(action.copy())
                    obs_list.append(obs)
                    # contact logging
                    contacts = p.getContactPoints(self.robot, self.cube)
                    counts.append(len(contacts))
                    forces.append(sum(c[9] for c in contacts))
                if attach:
                    self.attach_constraint()
            print(obs_list)
            print(act_list)
            all_obs.append(np.stack(obs_list, axis=0))
            all_act.append(np.stack(act_list, axis=0))
            all_counts.append(np.array(counts, dtype=int))
            all_forces.append(np.array(forces, dtype=float))
            print(f"Episode {ep+1}/{self.num_episodes} recorded.")

        # save
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, "data_obs.npy"), np.stack(all_obs, axis=0))
        np.save(os.path.join(self.save_dir, "data_act.npy"), np.stack(all_act, axis=0))
        np.save(os.path.join(self.save_dir, "data_contact_counts.npy"), np.stack(all_counts, axis=0))
        np.save(os.path.join(self.save_dir, "data_contact_forces.npy"), np.stack(all_forces, axis=0))
        print(f"Saved {self.num_episodes} episodes to '{self.save_dir}'")

if __name__ == "__main__":
    recorder = GraspRecorder(save_dir='data', num_episodes=50, gui=True)
    recorder.run()
"""