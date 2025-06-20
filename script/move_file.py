import os
import shutil

# ───────────────────────────────────────────
# 설정: 소스 폴더, 대상 폴더, 이동할 파일 목록
# ───────────────────────────────────────────
SRC_DIR  = r"C:\Users\Y\Imitation-Learning-for-Autonomous-Execution-of-Complex-Human-Workflows"        # 원본 CSV들이 있는 폴더
DEST_DIR = r"C:\Users\Y\Imitation-Learning-for-Autonomous-Execution-of-Complex-Human-Workflows\data\Pour\run50" # 파일을 옮길 목적지 폴더

FILES = [
    "robot_positions.csv",
    "cam0_timestamps.csv",
    "cam1_timestamps.csv",
    "matched_with_paths.csv"
]
# ───────────────────────────────────────────

def move_files(src_dir: str, dest_dir: str, filenames: list):
    # 대상 폴더가 없으면 생성
    os.makedirs(dest_dir, exist_ok=True)

    for name in filenames:
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            print(f"[SKIP] 파일이 없습니다: {src_path}")
            continue

        dst_path = os.path.join(dest_dir, name)
        try:
            shutil.move(src_path, dst_path)
            print(f"[OK]  {src_path} → {dst_path}")
        except Exception as e:
            print(f"[ERROR] {src_path} 이동 실패: {e}")

if __name__ == "__main__":
    move_files(SRC_DIR, DEST_DIR, FILES)
