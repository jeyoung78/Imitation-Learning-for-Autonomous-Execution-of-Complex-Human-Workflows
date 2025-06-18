import socket
import threading
import time
from datetime import datetime
from typing import Optional
from pypylon import pylon
import cv2
import os


global_target_x = 640
global_target_y = 408
global_margin = 15

def move(cx: int, cy: int, target_x=global_target_x, target_y=global_target_y, margin=global_margin):
    alignment = False
    if ((target_x - margin) <= cx <= (target_x + margin)) and ((target_y - margin) <= cy <= (target_y + margin)):
        alignment = True
    else:
        server.move_delta(cx=target_x-cx, cy=target_y-cy)

    return alignment

# — RobotServer 클래스 (변경 없음) —
class RobotServer:
    def __init__(self, host="0.0.0.0", port=20002, bufsize=1024, csv_path="robot_positions.csv"):
        self.host, self.port, self.bufsize, self.csv_path = host, port, bufsize, csv_path
        self._srv: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[RobotServer] Listening on {self.host}:{self.port}")
        self.conn, addr = srv.accept()
        print(f"[RobotServer] Connected by {addr}")
        self._srv = srv

    def send(self, msg: str) -> None:
        if not self.conn:
            raise RuntimeError("Connection not established. Call start() first.")
        self.conn.sendall(msg.encode("utf-8"))

    def receive(self) -> Optional[str]:
        if not self.conn:
            raise RuntimeError("Connection not established. Call start() first.")
        data = self.conn.recv(self.bufsize)
        if not data:
            return None
        text = data.decode("utf-8").strip()
        return text

    def close(self) -> None:
        if self.conn:
            self.conn.close()
        if self._srv:
            self._srv.close()

    def move_delta(self, cx: int, cy: int):
        delta_x = str(int(cx*0.3))
        delta_y = str(int(cy*0.3))
        
        self.send("x")
        time.sleep(0.1)
        self.send(delta_x)
        print("move x")
        self.rbt_wait()
        self.send("y")
        time.sleep(0.1)
        self.send(delta_y)
        print("move y")
        self.rbt_wait()

    def rbt_wait(self):
        print("wait for finish signal...")
        while True:
            robot_msg = self.receive()
            if robot_msg == 'finish':
                break
            elif robot_msg == 'no':
                print("no action in robot")
                break
            else:
                continue
        print("start next action.")
        time.sleep(1)

    def receive_loop(self):
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("timestamp_ms,x,y,z,w,p,r\n")
            while True:
                try:
                    data = self.conn.recv(self.bufsize)
                    if not data: break
                except ConnectionAbortedError:
                    print("[RobotServer] Connection aborted.")
                    break
                text = data.decode("utf-8", errors="ignore").strip()
                parts = text.split(",")
                if len(parts) != 6: continue
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f".{datetime.now().microsecond//1000:03d}"
                f.write(f"{ts}," + ",".join(parts) + "\n")
                f.flush()
                print(f"[RobotServer] {ts}, " + ",".join(parts))
        self.conn.close()

    def close(self):
        if self._srv: self._srv.close()


# — 카메라 녹화 함수 (640×408 리사이즈 적용) —
DURATION = 65
TARGET_FPS = 8.0
EXPECTED_FRAMES = int(DURATION * TARGET_FPS)
TARGET_W, TARGET_H = 640, 408

def record_camera(idx, cam, video_dir=r"D:\test_image"):
    os.makedirs(video_dir, exist_ok=True)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # 첫 프레임 Grab (크기 확인용)
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    res0 = cam.RetrieveResult(5000, pylon.TimeoutHandling_Return)
    frame0 = converter.Convert(res0).GetArray()
    res0.Release()

    # 비디오 라이터: 리사이즈된 크기로
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_path = os.path.join(video_dir, f"cam{idx}.avi")
    out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (TARGET_W, TARGET_H))
    print(f"[Camera {idx}] 녹화 시작: {datetime.now().strftime('%H:%M:%S.%f')[:-3]} → {video_path}")

    # 타임스탬프 로그
    log_path = f"cam{idx}_timestamps.csv"
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("frame,timestamp\n")
        frame_count = 0
        last_frame = frame0
        end_time = time.time() + DURATION

        while time.time() < end_time:
            res = cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if res.GrabSucceeded():
                raw = converter.Convert(res).GetArray()
                last_frame = raw
                # 리사이즈
                img = cv2.resize(raw, (TARGET_W, TARGET_H))
                out.write(img)
                frame_count += 1
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_f.write(f"{frame_count},{ts}\n")
                print(f"[Camera {idx}] Frame {frame_count:03d} at {ts[-12:]}")
            res.Release()

        # 부족분 채우기
        while frame_count < EXPECTED_FRAMES:
            img = cv2.resize(last_frame, (TARGET_W, TARGET_H))
            out.write(img)
            frame_count += 1
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_f.write(f"{frame_count},{ts}\n")
            print(f"[Camera {idx}] (복제) Frame {frame_count:03d} at {ts[-12:]}")

    out.release()
    cam.StopGrabbing()
    cam.Close()
    print(f"[Camera {idx}] 녹화 종료: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"[Camera {idx}] SUCCESS: {video_path}")
    print(f"[Camera {idx}] timestamps saved: {log_path}")


# — 메인 실행부 —
if __name__ == "__main__":
    # 1) 로봇 서버
    server = RobotServer()
    server.start()
    t_robot = threading.Thread(target=server.receive_loop, daemon=True)
    t_robot.start()

    # 2) 카메라 인스턴스 생성
    device_infos = pylon.TlFactory.GetInstance().EnumerateDevices()
    cams = []
    for dev_info in device_infos[:2]:
        cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev_info))
        cam.Open()
        cams.append(cam)

    # 3) 카메라 녹화 스레드
    threads = []
    for i, cam in enumerate(cams):
        t = threading.Thread(target=record_camera, args=(i, cam), daemon=True)
        t.start()
        threads.append(t)


    for t in threads: t.join()
    server.close()
    print("전체 녹화 및 로봇 기록 완료.")


