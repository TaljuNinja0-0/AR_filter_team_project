import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
ear_path = "assets/bunny_filter_1.png"
nose_path = "assets/bunny_filter_2.png"
input_path = "smile_girl.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
bunny_ear_img = cv2.imread(ear_path, cv2.IMREAD_UNCHANGED)
bunny_nose_img = cv2.imread(nose_path, cv2.IMREAD_UNCHANGED)

prev_rvec = None
prev_tvec = None
prev_angles = None

# === 얼굴 3D 모델 포인트 ===
model_points = np.array([
    (0.0, 0.0, 0.0),           # 코 끝
    (0.0, -63.6, -12.0),       # 턱
    (-43.3, 32.7, -26.0),      # 왼쪽 눈 바깥
    (43.3, 32.7, -26.0),       # 오른쪽 눈 바깥
    (-28.9, -28.9, -24.1),     # 왼쪽 입
    (28.9, -28.9, -24.1)       # 오른쪽 입
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
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
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

def apply_bunny_filter(frame):
    global prev_rvec, prev_tvec, prev_angles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        pts = np.array([[p.x, p.y] for p in landmarks.parts()])

        # ===== 2D → 3D 매핑용 포인트 =====
        image_points = np.array([
            pts[30],  # nose tip
            pts[8],   # chin
            pts[36],  # left eye outer
            pts[45],  # right eye outer
            pts[48],  # left mouth corner
            pts[54]   # right mouth corner
        ], dtype=np.float32)

        h, w = frame.shape[:2]
        cam_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # ===== 안정적 solvePnP =====
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

        # ===== 오일러 각 안정화 =====
        pitch, yaw, roll = euler_from_rvec(rvec)
        if prev_angles is not None:
            prev_pitch, prev_yaw, prev_roll = prev_angles
            alpha = 0.7
            pitch = alpha * prev_pitch + (1 - alpha) * pitch
            yaw   = alpha * prev_yaw   + (1 - alpha) * yaw
            roll  = alpha * prev_roll  + (1 - alpha) * roll
        prev_angles = (pitch, yaw, roll)
        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

        # ===== 얼굴 구조 기준 =====
        eye_left_3D = model_points[2]
        eye_right_3D = model_points[3]
        eye_center_3D = (eye_left_3D + eye_right_3D) / 2
        nose_tip_3D = model_points[0]
        eye_distance_3D = np.linalg.norm(eye_right_3D - eye_left_3D)
        
        # ===== (1) 토끼 귀 3D 위치 =====
        ear_width = eye_distance_3D * 2.0
        ear_height = eye_distance_3D * 1.2
        ear_center_3D = nose_tip_3D + np.array([0, eye_distance_3D * 1.8, -eye_distance_3D * 0.3])
        ear_3D = np.array([
            [-ear_width/2,  ear_height/2, 0],
            [ ear_width/2,  ear_height/2, 0],
            [ ear_width/2, -ear_height/2, 0],
            [-ear_width/2, -ear_height/2, 0]
        ], dtype=np.float32) + ear_center_3D

        # ===== (2) 토끼 코 3D 위치 =====
        nose_size_3D = eye_distance_3D * 0.7
        #nose_center_3D = nose_tip_3D + np.array([0, -18, -6])

        offset_x = -2  # 좌우 조정
        offset_y = -eye_distance_3D * 0.2  # 코 아래쪽
        offset_z = -eye_distance_3D * 0.1  # 코 앞쪽
        nose_center_3D = nose_tip_3D + np.array([offset_x, offset_y, offset_z])

        nose_3D = np.array([
            [-nose_size_3D*1.2,  nose_size_3D/2, 0],
            [ nose_size_3D*1.2,  nose_size_3D/2, 0],
            [ nose_size_3D*1.2, -nose_size_3D/2, 0],
            [-nose_size_3D*1.2, -nose_size_3D/2, 0]
        ], dtype=np.float32) + nose_center_3D

        # ===== warp + alpha 합성 함수 =====
        def warp_and_overlay(filter_img, target_3D):
            if filter_img is None:
                return frame
            projected_pts, _ = cv2.projectPoints(target_3D, rvec, tvec, cam_matrix, dist_coeffs)
            dst_pts = projected_pts.reshape(-1, 2).astype(np.float32)

            alpha = filter_img[:, :, 3]
            ys, xs = np.where(alpha > 0)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = filter_img[y_min:y_max + 1, x_min:x_max + 1]
            h_, w_ = cropped.shape[:2]
            src_pts = np.array([
                [0, 0],
                [w_ - 1, 0],
                [w_ - 1, h_ - 1],
                [0, h_ - 1]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(cropped, H, (frame.shape[1], frame.shape[0]), borderValue=(0, 0, 0, 0))
            return overlay_transparent(frame, warped, 0, 0)

        # ===== 매핑 실행 =====
        frame = warp_and_overlay(bunny_ear_img, ear_3D)
        frame = warp_and_overlay(bunny_nose_img, nose_3D)

    return frame

# ====== 입력 처리 ======
ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    result = apply_bunny_filter(img)
    output_path = os.path.splitext(input_path)[0] + "_bunny.png"
    cv2.imwrite(output_path, result)
    cv2.imshow("AR Bunny (Image)", result)
    print(f"✅ 이미지 결과 저장 완료: {output_path}")
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
        cv2.imshow("AR Bunny (Video)", result)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")
