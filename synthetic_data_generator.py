import os
import argparse
import datetime
import numpy as np
import pandas as pd
from PIL import Image

def generate_episode(ep_dir, T, H, W, C):
    # create directories
    cam0_dir = os.path.join(ep_dir, 'frames_cam0')
    cam1_dir = os.path.join(ep_dir, 'frames_cam1')
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)

    # timestamp base
    base_ts = datetime.datetime.now()

    rows = []
    for t in range(T):
        # 1) generate 7-DoF joint vector
        joints = np.random.uniform(-1.0, 1.0, size=7).astype(np.float32)

        # 2) generate & save two random RGB images
        img0 = (np.random.rand(H, W, C) * 255).astype(np.uint8)
        img1 = (np.random.rand(H, W, C) * 255).astype(np.uint8)

        cam0_fname = f'cam0_{t:04d}.png'
        cam1_fname = f'cam1_{t:04d}.png'
        cam0_path = os.path.join(cam0_dir, cam0_fname)
        cam1_path = os.path.join(cam1_dir, cam1_fname)

        Image.fromarray(img0).save(cam0_path)
        Image.fromarray(img1).save(cam1_path)

        # 3) timestamps and frame indices
        robot_ts  = base_ts + datetime.timedelta(milliseconds=100*t)
        cam0_ts   = robot_ts
        cam1_ts   = robot_ts

        rows.append([
            robot_ts.isoformat(),
            *joints.tolist(),
            t,
            cam0_ts.isoformat(),
            t,
            cam1_ts.isoformat(),
            cam0_path,
            cam1_path
        ])

    # 4) write CSV
    cols = [
        'robot_ts',
        'x','y','z','w','p','r','ee',
        'cam0_frame','cam0_ts','cam1_frame','cam1_ts',
        'cam0_path','cam1_path'
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(os.path.join(ep_dir, 'matched_with_paths.csv'), index=False)


def main():
    p = argparse.ArgumentParser(
        description='Generate synthetic two-camera + 7-DoF episodes')
    p.add_argument('--output_dir', required=True,
                   help='where to write episode folders')
    p.add_argument('--num_episodes', type=int, default=5,
                   help='how many episodes to generate')
    p.add_argument('--timesteps',   type=int, default=100,
                   help='frames (and joint vectors) per episode')
    p.add_argument('--height',      type=int, default=128,
                   help='image height in pixels')
    p.add_argument('--width',       type=int, default=128,
                   help='image width in pixels')
    p.add_argument('--channels',    type=int, default=3,
                   help='number of image channels (3=RGB)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.num_episodes):
        ep_name = f'episode_{i:03d}'
        ep_dir  = os.path.join(args.output_dir, ep_name)
        generate_episode(ep_dir,
                         T       = args.timesteps,
                         H       = args.height,
                         W       = args.width,
                         C       = args.channels)
    print(f'✔ Generated {args.num_episodes} episodes in {args.output_dir}')


if __name__ == '__main__':
    main()