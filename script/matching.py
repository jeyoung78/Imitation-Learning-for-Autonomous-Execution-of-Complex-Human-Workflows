import pandas as pd
import os

BASE_DIR = r"D:\test_image\run50"

# 1) 파일 읽기
robot = pd.read_csv('robot_positions.csv', parse_dates=['timestamp_ms'])
cam0  = pd.read_csv('cam0_timestamps.csv',  parse_dates=['timestamp'])
cam1  = pd.read_csv('cam1_timestamps.csv',  parse_dates=['timestamp'])

# 2) 열명 정리
robot = robot.rename(columns={'timestamp_ms':'robot_ts'})
cam0  = cam0.rename (columns={'timestamp':'cam0_ts', 'frame':'cam0_frame'})
cam1  = cam1.rename (columns={'timestamp':'cam1_ts', 'frame':'cam1_frame'})

# 3) 정렬
robot = robot.sort_values('robot_ts')
cam0  = cam0.sort_values('cam0_ts')
cam1  = cam1.sort_values('cam1_ts')

# 4) asof 매칭
matched0 = pd.merge_asof(robot[['robot_ts']], cam0, left_on='robot_ts', right_on='cam0_ts', direction='nearest')
matched1 = pd.merge_asof(robot[['robot_ts']], cam1, left_on='robot_ts', right_on='cam1_ts', direction='nearest')

# 5) 로봇 테이블에 병합
df = robot.join(matched0[['cam0_frame','cam0_ts']]).join(matched1[['cam1_frame','cam1_ts']])

# 6) 이미지 경로 컬럼 추가
def make_path(cam_idx, frame_no):
    # 4자리 0패딩, 확장자 .png
    fname = f"cam{cam_idx}_{int(frame_no):04d}.png"
    return os.path.join(BASE_DIR, f"frames_cam{cam_idx}", fname)

df['cam0_path'] = df['cam0_frame'].apply(lambda i: make_path(0, i))
df['cam1_path'] = df['cam1_frame'].apply(lambda i: make_path(1, i))

# 7) 저장
df.to_csv('matched_with_paths.csv', index=False)
print("👉 matched_with_paths.csv 에 저장되었습니다.")
