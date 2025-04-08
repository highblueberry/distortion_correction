import cv2 as cv
import numpy as np

def extract_frames_and_detect_corners(video_path, board_pattern, interval):
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    valid_images = []
    image_size = None

    # Prepare 3D object points for the checkerboard
    objp = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)

    cap = cv.VideoCapture(video_path)
    assert cap.isOpened(), "비디오 파일을 열 수 없습니다."
    
    frame_idx = 0  # 프레임 번호 초기화

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # interval 단위로만 프레임 처리
        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 코너 검출 (옵션 추가로 더 잘 잡힘)
        found, corners = cv.findChessboardCorners(
            gray, board_pattern,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )

        print(f"[{frame_idx}] 코너 찾았나? => {found}")

        if found:
            print(f"[{frame_idx}] 코너 검출 성공")

            if image_size is None:
                image_size = gray.shape[::-1]

            # 코너 정밀화 및 추가
            corners2 = cv.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_images.append(frame.copy())  # 보정할 원본 이미지 저장

            # 시각화 (원하면 꺼도 됨)
            cv.drawChessboardCorners(frame, board_pattern, corners2, found)
            cv.imshow("Corners", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        frame_idx += 1  # 꼭 증가시켜줘야 함!

    cap.release()
    cv.destroyAllWindows()

    return objpoints, imgpoints, image_size, valid_images


def calibrate_camera(objpoints, imgpoints, image_size):
    if not objpoints or not imgpoints:
        raise RuntimeError("코너를 검출한 프레임이 없습니다. 캘리브레이션 실패.")

    # 카메라 캘리브레이션
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return ret, K, dist, rvecs, tvecs


if __name__ == '__main__':
    # === 설정 ===
    video_path = "data\chessboard.avi"  # 비디오 경로
    board_pattern = (10, 7)  # 내부 코너 수 (가로 x 세로) 
    frame_interval = 80
    board_cellsize = 0.025  # 1칸 실제 크기 (예: 2.5cm)

    print("체스보드 인식 중...")
    objpoints, imgpoints, image_size, valid_images = extract_frames_and_detect_corners(video_path, board_pattern, interval=frame_interval)

    print("카메라 캘리브레이션 수행...")
    rms, K, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)

    print("\n=== 캘리브레이션 결과 ===")
    print(f"* 사용된 프레임 수: {len(objpoints)}")
    print(f"* RMS error: {rms:.4f}")
    print(f"* Camera matrix (K):\n{K}")
    print(f"* Distortion coefficients:\n{dist.ravel()}")
    
    
    # === 왜곡 보정 및 저장 ===
    print("\n💾 왜곡 보정된 이미지 저장 중...")
    for i, img in enumerate(valid_images):
        undistorted = cv.undistort(img, K, dist)
        combined = np.hstack((img, undistorted))  # 원본 + 보정된 이미지 나란히
        filename = f"data\compare_undistorted_{i:02d}.png"
        cv.imwrite(filename, combined)
        print(f"🖼️ 저장 완료: {filename}")