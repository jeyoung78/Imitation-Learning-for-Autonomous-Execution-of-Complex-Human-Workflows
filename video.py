'''
from pypylon import pylon
import cv2, threading, time, datetime

DURATION = 15 # 녹화 시간(초)

# Basler 카메라 2대 열기
devices = pylon.TlFactory.GetInstance().EnumerateDevices()
if len(devices) < 2:
    raise RuntimeError("Basler 카메라 2대 필요")

cams = []
for dev in devices[:2]:
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev))
    cam.Open()

    # ROI 풀 리셋
    try:
        cam.OffsetX.SetValue(0)
        cam.OffsetY.SetValue(0)
        cam.Width.   SetValue(cam.Width.   GetMax())
        cam.Height.  SetValue(cam.Height.  GetMax())
    except:
        pass

    cams.append(cam)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat  = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

def record_camera(idx, cam):
    # 첫 프레임 받아서 크기 확인
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    res = cam.RetrieveResult(5000, pylon.TimeoutHandling_Return)
    if not res.GrabSucceeded():
        print(f"[ERROR] Cam{idx} 초기 Grab 실패")
        cam.StopGrabbing()
        return

    img0 = converter.Convert(res).GetArray()
    h, w = img0.shape[:2]
    res.Release()

    # ▶ 고정 FPS 사용
    fps = 10.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'cam{idx}_output.avi', fourcc, fps, (w, h))

    start_ts = datetime.datetime.now()
    print(f"[Camera {idx}] 녹화 시작: {start_ts:%Y-%m-%d %H:%M:%S}")

    # 첫 프레임 저장
    out.write(img0)

    # 나머지 프레임 녹화
    end_time = time.time() + DURATION
    while time.time() < end_time:
        res = cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
        if not res.GrabSucceeded():
            res.Release()
            continue
        img = converter.Convert(res).GetArray()
        res.Release()
        out.write(img)

    cam.StopGrabbing()
    end_ts = datetime.datetime.now()
    print(f"[Camera {idx}] 녹화 종료: {end_ts:%Y-%m-%d %H:%M:%S}")

    out.release()
    cam.Close()

threads = []
for i, cam in enumerate(cams):
    t = threading.Thread(target=record_camera, args=(i, cam), daemon=True)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

cv2.destroyAllWindows()
print("전체 녹화 완료")
'''
#!/usr/bin/env python3
from pypylon import pylon
import cv2
import time
import datetime
import argparse

# ------------------------------------------------------------------------------
# CLI 인자 파싱
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Record from a single Basler camera for a fixed duration & FPS"
)
parser.add_argument(
    "--cam", type=int, required=True,
    help="Which camera index to use (e.g. 0 or 1)"
)
args = parser.parse_args()

# ------------------------------------------------------------------------------
# 녹화 설정
# ------------------------------------------------------------------------------
DURATION = 30.0   # seconds
FPS      = 7.5

# ------------------------------------------------------------------------------
# 연결된 Basler 카메라 열기
# ------------------------------------------------------------------------------
devices = pylon.TlFactory.GetInstance().EnumerateDevices()
if args.cam < 0 or args.cam >= len(devices):
    raise RuntimeError(
        f"Camera index {args.cam} is invalid. Found {len(devices)} device(s)."
    )

cam = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateDevice(devices[args.cam])
)
cam.Open()

# (선택) ROI 풀 리셋
try:
    cam.OffsetX.SetValue(0)
    cam.OffsetY.SetValue(0)
    cam.Width.SetValue(cam.Width.GetMax())
    cam.Height.SetValue(cam.Height.GetMax())
except Exception:
    pass

# ------------------------------------------------------------------------------
# 이미지 컨버터 & 연속 모드 설정
# ------------------------------------------------------------------------------
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat   = pylon.PixelType_BGR8packed
converter.OutputBitAlignment  = pylon.OutputBitAlignment_MsbAligned

# 최신 프레임만 가져오도록 continuous grab
cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# ------------------------------------------------------------------------------
# 첫 프레임으로 해상도 확인 및 VideoWriter 생성
# ------------------------------------------------------------------------------
res = cam.RetrieveResult(5000, pylon.TimeoutHandling_Return)
if not res.GrabSucceeded():
    raise RuntimeError("첫 프레임 Grab 실패")
img0 = converter.Convert(res).GetArray()
h, w = img0.shape[:2]
res.Release()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    f'cam{args.cam}_output.avi',
    fourcc, FPS, (w, h)
)

# ------------------------------------------------------------------------------
# 녹화 시작 로그 & 타이밍 설정
# ------------------------------------------------------------------------------
start_perf = time.perf_counter()
start_dt   = datetime.datetime.now()
end_perf   = start_perf + DURATION
next_frame = start_perf

print(f"[Camera {args.cam}] 녹화 시작: {start_dt:%Y-%m-%d %H:%M:%S}")

# ------------------------------------------------------------------------------
# 녹화 루프
# ------------------------------------------------------------------------------
# 첫 프레임 기록
out.write(img0)
next_frame += 1.0 / FPS

while time.perf_counter() < end_perf:
    now = time.perf_counter()
    if now < next_frame:
        time.sleep(next_frame - now)

    res = cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
    if not res.GrabSucceeded():
        res.Release()
        next_frame += 1.0 / FPS
        continue

    img = converter.Convert(res).GetArray()
    res.Release()
    out.write(img)
    next_frame += 1.0 / FPS

# ------------------------------------------------------------------------------
# 정리 및 종료 로그
# ------------------------------------------------------------------------------
cam.StopGrabbing()
out.release()
cam.Close()

end_dt = start_dt + datetime.timedelta(seconds=DURATION)
print(f"[Camera {args.cam}] 녹화 종료: {end_dt:%Y-%m-%d %H:%M:%S}")
