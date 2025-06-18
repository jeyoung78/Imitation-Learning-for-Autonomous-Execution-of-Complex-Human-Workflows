import cv2
import os

def split_video_to_frames(video_path, output_dir, prefix="frame"):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"열린 영상: {video_path} ({fps:.2f} FPS, {total_frames} frames)")

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 파일명: frame_0001.png, frame_0002.png, ...
        out_path = os.path.join(output_dir, f"{prefix}_{frame_idx:04d}.png")
        cv2.imwrite(out_path, frame)
        print(f"[{video_path}] 프레임 {frame_idx:04d} 저장 → {out_path}")
        frame_idx += 1

    cap.release()
    print(f"완료: 총 {frame_idx} 프레임을 {output_dir}에 저장했습니다.")

if __name__ == "__main__":
    # 예시: cam0.avi, cam1.avi를 분할
    split_video_to_frames(r"D:\test_image\cam0.avi", r"D:\test_image\frames_cam0", prefix="cam0")
    split_video_to_frames(r"D:\test_image\cam1.avi", r"D:\test_image\frames_cam1", prefix="cam1")
