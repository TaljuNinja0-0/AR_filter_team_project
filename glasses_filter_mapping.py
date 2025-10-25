import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets/glasses_filter.png"
input_path = "woman.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

prev_rvec = None
prev_tvec = None
prev_angles = None

# === 기본 3D 모델 포인트 (단위: mm) ===
model_points = np.array([
    (0.0, 0.0, 0.0),             # 코 끝
    (0.0, -63.6, -12.0),         # 턱
    (-43.3, 32.7, -26.0),        # 왼쪽 눈 바깥
    (43.3, 32.7, -26.0),         # 오른쪽 눈 바깥
    (-28.9, -28.9, -24.1),       # 왼쪽 입
    (28.9, -28.9, -24.1)         # 오른쪽 입
])

# === 카메라 매트릭스 (대략적인 값) ===
focal_length = 1
center = (0.5, 0.5)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)
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

    return np.degrees(x), np.degrees(y), np.degrees(z)  # pitch, yaw, roll


def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
        return background

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)
    overlay_x1 = max(-x, 0)
    overlay_y1 = max(-y, 0)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    background_roi = background[y1:y2, x1:x2]
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_crop.shape[2] < 4:
        return background

    overlay_img = overlay_crop[..., :3]
    mask = overlay_crop[..., 3:] / 255.0
    blended = (1 - mask) * background_roi + mask * overlay_img
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def curve_warp_glasses(img, yaw=0.0, pitch=0.0):
    """yaw(좌우 회전)과 pitch(상하) 기반으로 곡률을 다르게 줌"""
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    center_x = w / 2

    # 곡률 계산 (yaw, pitch 기반)
    base_curv = 0.05
    yaw_factor = min(abs(yaw) / 25.0, 1.0)  # 최대 ±25도 기준
    pitch_factor = min(abs(pitch) / 20.0, 1.0)
    left_curvature = base_curv + 0.05 * yaw_factor
    right_curvature = base_curv + 0.05 * yaw_factor
    if yaw > 0:  # 오른쪽 회전 시 오른쪽 곡률을 더 줄임
        right_curvature *= 0.7
    elif yaw < 0:
        left_curvature *= 0.7

    for y in range(h):
        for x in range(w):
            if x < center_x:
                offset = left_curvature * ((x - center_x) ** 2) / (center_x ** 2)
            else:
                offset = right_curvature * ((x - center_x) ** 2) / (center_x ** 2)
            map_x[y, x] = x
            map_y[y, x] = y + offset * h * 0.25  # 완화된 곡률

    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return warped


def apply_glasses_filter(frame):
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
        cam_matrix = np.array([[w, 0, w / 2],
                               [0, w, h / 2],
                               [0, 0, 1]], dtype=np.float32)

        # 안정적인 solvePnP 호출
        if prev_rvec is None:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, cam_matrix, dist_coeffs,
                rvec=prev_rvec, tvec=prev_tvec,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            continue

        # 오일러 각 계산
        pitch, yaw, roll = euler_from_rvec(rvec)

        # ---- (1) yaw 연속성 보장 ----
        if prev_angles is not None:
            prev_pitch, prev_yaw, prev_roll = prev_angles
            # yaw가 갑자기 150도 이상 튀면 방향 반전된 것으로 간주
            if abs(yaw - prev_yaw) > 150:
                yaw = -yaw
                pitch = -pitch
                roll = -roll
                rvec = -rvec

            # 스무딩 적용
            alpha = 0.7
            pitch = alpha * prev_pitch + (1 - alpha) * pitch
            yaw   = alpha * prev_yaw + (1 - alpha) * yaw
            roll  = alpha * prev_roll + (1 - alpha) * roll

        prev_angles = (pitch, yaw, roll)
        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

        # ====== 안경 회전 및 스케일 ======
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        eye_width = np.linalg.norm(right_eye - left_eye)
        scale = (eye_width / filter_img.shape[1]) * 3.0

        new_w = int(filter_img.shape[1] * scale)
        new_h = int(filter_img.shape[0] * scale)
        resized = cv2.resize(filter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # yaw 보정 (뒤집힘 방지)
        angle = -roll  # roll만 사용 (yaw/pitch는 warp로 처리)
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1)
        rotated = cv2.warpAffine(resized, M, (new_w, new_h), borderValue=(0, 0, 0, 0))

        # 곡률 보정
        curved = curve_warp_glasses(rotated, yaw=yaw, pitch=pitch)

        face_center = pts[27]
        x = int(face_center[0] - curved.shape[1] / 2)
        y = int(face_center[1] - curved.shape[0] / 2.5)

        frame = overlay_transparent(frame, curved, x, y)

    return frame

ext = os.path.splitext(input_path)[1].lower()

if ext in [".jpg", ".jpeg", ".png"]:
    # --- 이미지 입력 ---
    img = cv2.imread(input_path)
    result = apply_glasses_filter(img)
    output_path = os.path.splitext(input_path)[0] + "_output.png"
    cv2.imwrite(output_path, result)
    cv2.imshow("AR Glasses (Image)", result)
    print(f"✅ 이미지 결과 저장 완료: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # --- 영상 입력 ---
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_output.mp4"
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_glasses_filter(frame)
        out.write(result)
        cv2.imshow("AR Glasses (Video)", result)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")
