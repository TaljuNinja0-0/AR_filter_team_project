import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets/bunny_filter_1.png"
input_path = "Lena.png"  # 또는 이미지 파일

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# === 필터 읽고 상하 뒤집기 ===
filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
filter_img = cv2.flip(filter_img, 0)  # 상하 뒤집기

# 필터 여백 제거
if filter_img is not None and filter_img.shape[2] == 4:
    alpha = filter_img[:, :, 3]
    ys, xs = np.where(alpha > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    filter_img = filter_img[y_min:y_max + 1, x_min:x_max + 1]

prev_rvec, prev_tvec, prev_angles = None, None, None

# === 3D 얼굴 모델 포인트 ===
model_points = np.array([
    (0.0, 0.0, 0.0),           # 코 끝
    (0.0, -63.6, -12.0),       # 턱
    (-43.3, 32.7, -26.0),      # 왼쪽 눈 바깥
    (43.3, 32.7, -26.0),       # 오른쪽 눈 바깥
    (-28.9, -28.9, -24.1),     # 왼쪽 입
    (28.9, -28.9, -24.1)       # 오른쪽 입
])
dist_coeffs = np.zeros((4, 1))


def euler_from_rvec(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)


def overlay_transparent(background, overlay, x=0, y=0):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]
    if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
        return background

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)

    overlay_x1, overlay_y1 = max(-x, 0), max(-y, 0)
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    if overlay_crop.shape[2] < 4:
        return background

    background_roi = background[y1:y2, x1:x2]
    overlay_img = overlay_crop[..., :3]
    mask = overlay_crop[..., 3:] / 255.0
    blended = (1 - mask) * background_roi + mask * overlay_img
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background

def apply_bunny_filter(frame):
    global prev_rvec, prev_tvec, prev_angles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        pts = np.array([[p.x, p.y] for p in landmarks.parts()])

        # SolvePnP용 2D 포인트
        image_points = np.array([
            pts[30],  # 코 끝
            pts[8],   # 턱
            pts[36],  # 왼쪽 눈 바깥
            pts[45],  # 오른쪽 눈 바깥
            pts[48],  # 왼쪽 입
            pts[54]   # 오른쪽 입
        ], dtype=np.float32)

        h, w = frame.shape[:2]
        cam_matrix = np.array([[w, 0, w/2],
                               [0, w, h/2],
                               [0, 0, 1]], dtype=np.float32)

        # 안정적인 pose estimation
        if prev_rvec is None:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, cam_matrix, dist_coeffs,
                rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not success:
            continue

        # 오일러 각 계산
        pitch, yaw, roll = euler_from_rvec(rvec)

        # 각도 보정 (연속성)
        if prev_angles is not None:
            prev_pitch, prev_yaw, prev_roll = prev_angles
            alpha = 0.7
            pitch = alpha * prev_pitch + (1 - alpha) * pitch
            yaw = alpha * prev_yaw + (1 - alpha) * yaw
            roll = alpha * prev_roll + (1 - alpha) * roll
        prev_angles = (pitch, yaw, roll)
        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

        # ====== 귀 위치 기준점 계산 ======
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        forehead_center = (left_eye + right_eye) / 2

        # 귀를 머리 꼭대기로 이동 (기존보다 훨씬 위)
        forehead_center[1] -= face.height() * 1.1

        # ====== 귀 크기 계산 (조정됨) ======
        face_width = face.right() - face.left()
        scale = face_width / filter_img.shape[1] * 1.8  # 크기 2배 키움
        new_w = int(filter_img.shape[1] * scale)
        new_h = int(filter_img.shape[0] * scale)
        resized = cv2.resize(filter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ====== 3D 귀 포인트 정의 (상하 반전 적용됨) ======
        ear_3D = np.array([
            [-new_w/2, -new_h/2, 0],
            [ new_w/2, -new_h/2, 0],
            [ new_w/2,  new_h/2, 0],
            [-new_w/2,  new_h/2, 0]
        ], dtype=np.float32)

        # 투영
        projected_pts, _ = cv2.projectPoints(ear_3D, rvec, tvec, cam_matrix, dist_coeffs)
        projected_pts = projected_pts.reshape(-1,2).astype(np.float32)

        # 중심 이동 (이마 기준)
        proj_center = projected_pts.mean(axis=0)
        projected_pts += (forehead_center - proj_center)

        # ====== 투영 및 합성 ======
        src_pts = np.array([[0,0],[new_w-1,0],[new_w-1,new_h-1],[0,new_h-1]], dtype=np.float32)
        dst_pts = projected_pts
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(resized, H, (frame.shape[1], frame.shape[0]), borderValue=(0,0,0,0))
        frame = overlay_transparent(frame, warped, 0, 0)

    return frame


# ====== 입력 처리 ======
ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    result = apply_bunny_filter(img)
    out_path = os.path.splitext(input_path)[0] + "_bunny.png"
    cv2.imwrite(out_path, result)
    cv2.imshow("AR Bunny (3D)", result)
    print(f"✅ 이미지 결과 저장 완료: {out_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_bunny.mp4"
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = apply_bunny_filter(frame)
        out.write(result)
        cv2.imshow("AR Bunny (3D Video)", result)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")
