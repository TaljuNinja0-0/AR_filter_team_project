import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
whitemask_path = "assets/whitemask_filter.png"
#input_path = "smile_girl.mp4"  # "Lena.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
whitemask_img = cv2.imread(whitemask_path, cv2.IMREAD_UNCHANGED)

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



def apply_whitemask_filter(frame):
    global prev_rvec, prev_tvec, prev_angles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        pts = np.array([[p.x, p.y] for p in landmarks.parts()])

        # ===== 얼굴 기준 포인트 =====
        forehead = pts[27].astype(np.float32)
        chin = pts[8].astype(np.float32)
        jaw_left = pts[0].astype(np.float32)   # 왼쪽 얼굴 끝
        jaw_right = pts[16].astype(np.float32) # 오른쪽 얼굴 끝
        # ===== 가면 중심 & 크기 =====
        mask_center = pts[28]
        mask_width = int(np.linalg.norm(jaw_right - jaw_left) * 2.8)  # 약간 여유
        mask_height = int(np.linalg.norm(chin - forehead) * 2.8)  

        # ===== 각도 계산 =====
        dx = jaw_right[0] - jaw_left[0]
        dy = jaw_right[1] - jaw_left[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # ===== PNG 자르기 & 크기 조절 =====
        alpha = whitemask_img[:, :, 3]
        ys, xs = np.where(alpha > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        mask_cropped = whitemask_img[y_min:y_max + 1, x_min:x_max + 1]
        resized = cv2.resize(mask_cropped, (mask_width, mask_height), interpolation=cv2.INTER_AREA)

        # ===== 회전 =====
        M = cv2.getRotationMatrix2D((mask_width / 2, mask_height / 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (mask_width, mask_height), borderValue=(0, 0, 0, 0))

        # ===== 위치 계산 & 합성 =====
        x1 = int(mask_center[0] - mask_width / 2)
        y1 = int(mask_center[1] - mask_height / 2)
        frame = overlay_transparent(frame, rotated, x1, y1)

    return frame


'''
# ====== 입력 처리 ======
ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    result = apply_whitemask_filter(img)
    output_path = os.path.splitext(input_path)[0] + "_whitemask.png"
    cv2.imwrite(output_path, result)
    cv2.imshow("AR mask (Image)", result)
    print(f"✅ 이미지 결과 저장 완료: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_whitemask.mp4"
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = apply_whitemask_filter(frame)
        out.write(result)
        cv2.imshow("AR WhiteMask (Video)", result)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")
'''