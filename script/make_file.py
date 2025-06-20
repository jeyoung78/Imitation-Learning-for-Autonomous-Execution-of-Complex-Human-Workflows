# create_runs_under_pour.py

import os

def make_runs_under_pour(pour_dir: str, num_runs: int = 50):
    """
    pour_dir 안에 run1, run2, ..., run{num_runs} 폴더를 생성합니다.
    """
    for i in range(1, num_runs + 1):
        run_folder = os.path.join(pour_dir, f"run{i}")
        os.makedirs(run_folder, exist_ok=True)
        print(f"✔ Created: {run_folder}")

if __name__ == "__main__":
    # 실제 Pour 폴더 경로로 수정하세요
    POUR_DIR = r"C:\Users\Y\Imitation-Learning-for-Autonomous-Execution-of-Complex-Human-Workflows\data\Pour"
    make_runs_under_pour(POUR_DIR, num_runs=50)
