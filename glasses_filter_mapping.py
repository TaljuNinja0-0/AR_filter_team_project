import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets/glasses_filter.png"
input_path = "smile_girl.mp4" #"Lena.png"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

prev_rvec = None
prev_tvec = None
prev_angles = None

# === 기본 3D 모델 포인트 (단위: mm) ===
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
        cam_matrix = np.array([[w, 0, w/2],[0, w, h/2],[0,0,1]], dtype=np.float32)

        # 안정적인 solvePnP 호출
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

        # yaw 연속성 보정
        if prev_angles is not None:
            prev_pitch, prev_yaw, prev_roll = prev_angles
            if abs(yaw - prev_yaw) > 150:
                yaw = -yaw
                pitch = -pitch
                roll = -roll
                rvec = -rvec
            alpha = 0.7
            pitch = alpha * prev_pitch + (1 - alpha) * pitch
            yaw = alpha * prev_yaw + (1 - alpha) * yaw
            roll = alpha * prev_roll + (1 - alpha) * roll

        prev_angles = (pitch, yaw, roll)
        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

        # ====== 안경 크기/좌표 계산 ======
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        eye_center = (left_eye + right_eye) / 2
        eye_width = np.linalg.norm(right_eye - left_eye)

        # 안경 이미지 알파 채널 기준 크롭
        alpha = filter_img[:, :, 3]
        ys, xs = np.where(alpha > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        glasses_cropped = filter_img[y_min:y_max+1, x_min:x_max+1]
        g_h, g_w = glasses_cropped.shape[:2]
        content_w, content_h = x_max - x_min, y_max - y_min

        desired_ratio = 2.0
        scale_factor = (eye_width * desired_ratio) / g_w
        new_w = int(g_w * scale_factor)
        new_h = int(g_h * scale_factor)
        new_w = max(30, min(new_w, frame.shape[1]//2))  # 최소 30픽셀, 최대 화면 절반
        new_h = max(15, min(new_h, frame.shape[0]//4))  # 최소 15픽셀, 최대 화면 4분의 1
        resized = cv2.resize(glasses_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ====== 3D 안경 모델 포인트 정의 ======
        glasses_3D = np.array([
            [-new_w/2,  new_h/2, 0],  # 왼쪽 상단
            [ new_w/2,  new_h/2, 0],  # 오른쪽 상단
            [ new_w/2, -new_h/2, 0],  # 오른쪽 하단
            [-new_w/2, -new_h/2, 0]   # 왼쪽 하단
        ], dtype=np.float32)

        # SolvePnP 결과로 2D 투영
        eye_center_2D = (left_eye + right_eye) / 2
        projected_pts, _ = cv2.projectPoints(glasses_3D, rvec, tvec, cam_matrix, dist_coeffs)
        projected_pts = projected_pts.reshape(-1,2).astype(np.float32)

        proj_center = projected_pts.mean(axis=0)
        pixel_offset_y = new_h * 0.3  # 여기서는 양수 = 아래로 이동
        projected_pts += (eye_center_2D - proj_center) + np.array([0, pixel_offset_y])

        # 원본 안경 이미지 4코너
        src_pts = np.array([[0,0],[new_w-1,0],[new_w-1,new_h-1],[0,new_h-1]], dtype=np.float32)
        dst_pts = projected_pts

        # 호모그래피 + 투영
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(resized, H, (frame.shape[1], frame.shape[0]), borderValue=(0,0,0,0))
        frame = overlay_transparent(frame, warped, 0, 0)

    return frame

# ====== 입력 처리 ======
ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    result = apply_glasses_filter(img)
    output_path = os.path.splitext(input_path)[0] + "_output.png"
    cv2.imwrite(output_path, result)
    cv2.imshow("AR Glasses (Image)", result)
    print(f"✅ 이미지 결과 저장 완료: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
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
