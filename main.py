# launcher.py
import multiprocessing
import sys

def main(cam_index):
    # this is equivalent to running:
    #   python record_camera.py --cam cam_index
    # so we simply exec the same script in-process:
    import subprocess
    subprocess.run([sys.executable, "video.py", "--cam", str(cam_index)])

if __name__ == "__main__":
    procs = []
    for cam in (0, 1):
        p = multiprocessing.Process(target=main, args=(cam,))
        p.start()
        print(f"Started camera {cam} with PID {p.pid}")
        procs.append(p)

    for p in procs:
        p.join()

    print("Both recordings complete.")
