import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
santa_hat_path = "assets/santa_filter_1.png"
santa_beard_path = "assets/santa_filter_2.png"
input_path = "smile_girl3.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
santa_hat_img = cv2.imread(santa_hat_path, cv2.IMREAD_UNCHANGED)
santa_beard_img = cv2.imread(santa_beard_path, cv2.IMREAD_UNCHANGED)

# === 3D 얼굴 모델 포인트 (단위: mm) ===
model_points = np.array([
    (0.0, 0.0, 0.0),           # 코 끝
    (0.0, -330.0, -65.0),      # 턱
    (-225.0, 170.0, -135.0),   # 왼쪽 눈 바깥
    (225.0, 170.0, -135.0),    # 오른쪽 눈 바깥
    (-150.0, -150.0, -125.0),  # 왼쪽 입
    (150.0, -150.0, -125.0)    # 오른쪽 입
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))

# ===== PNG 크롭 캐싱 (한 번만 실행) =====
hat_cropped_cache = None
beard_cropped_cache = None

# ===== 크기 고정 변수 =====
fixed_hat_size = None
fixed_beard_size = None
is_size_initialized = False

def crop_transparent_region(img):
    """투명 영역 제거"""
    alpha = img[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0 or len(xs) == 0:
        return img
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return img[y_min:y_max + 1, x_min:x_max + 1]

def overlay_transparent(background, overlay, x, y):
    """투명 PNG 합성 (최적화)"""
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

    if overlay_crop.shape[2] < 4 or overlay_crop.size == 0:
        return background

    overlay_img = overlay_crop[..., :3]
    mask = overlay_crop[..., 3:] / 255.0
    blended = (1 - mask) * background_roi + mask * overlay_img
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def apply_santa_filter_optimized(frame):
    """최적화된 산타 필터 - 첫 프레임 크기 고정"""
    global hat_cropped_cache, beard_cropped_cache
    global fixed_hat_size, fixed_beard_size, is_size_initialized
    
    # 캐시 초기화 (한 번만)
    if hat_cropped_cache is None:
        hat_cropped_cache = crop_transparent_region(santa_hat_img)
    if beard_cropped_cache is None:
        beard_cropped_cache = crop_transparent_region(santa_beard_img)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ===== 얼굴 검출 =====
    faces = detector(gray, 0)

    if len(faces) == 0:
        return frame

    for face in faces:
        landmarks = predictor(gray, face)
        pts = np.array([[p.x, p.y] for p in landmarks.parts()])

        # ===== 2D 이미지 포인트 =====
        image_points = np.array([
            pts[30],  # 코 끝
            pts[8],   # 턱
            pts[36],  # 왼쪽 눈 바깥
            pts[45],  # 오른쪽 눈 바깥
            pts[48],  # 왼쪽 입
            pts[54]   # 오른쪽 입
        ], dtype=np.float32)

        # ===== 카메라 매트릭스 =====
        h, w = frame.shape[:2]
        cam_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # ===== solvePnP =====
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            continue

        # ===== 얼굴 기준 3D 좌표 =====
        eye_left_3D = model_points[2]
        eye_right_3D = model_points[3]
        eye_center_3D = (eye_left_3D + eye_right_3D) / 2
        eye_distance_3D = np.linalg.norm(eye_right_3D - eye_left_3D)
        face_width_3D = eye_distance_3D * 2.2
        face_height_3D = np.linalg.norm(model_points[1] - eye_center_3D)
        
        # ===== 첫 프레임에서만 크기 계산 =====
        if not is_size_initialized:
            nose_3D = model_points[0]
            chin_3D = model_points[1]
            
            # 크기 고정
            fixed_hat_size = (face_width_3D * 1.2, face_height_3D * 1.7)
            fixed_beard_size = (face_width_3D * 0.8, np.linalg.norm(chin_3D - nose_3D) * 1.8)
            
            is_size_initialized = True
            print(f"✅ 크기 고정: 모자({fixed_hat_size[0]:.1f}x{fixed_hat_size[1]:.1f}), 수염({fixed_beard_size[0]:.1f}x{fixed_beard_size[1]:.1f})")
        
        # ===== 고정된 크기 사용 =====
        hat_w_3D, hat_h_3D = fixed_hat_size
        beard_w_3D, beard_h_3D = fixed_beard_size
        
        # ===== 거리 기반 스케일 =====
        z_scale = max(0.5, min(2.0, 600 / (tvec[2] + 1e-5)))
        
        # ===== 산타 모자 =====
        hat_center_3D = eye_center_3D + np.array([eye_distance_3D * 0.25, face_height_3D * 0.5, -eye_distance_3D * 0.2])
        
        hat_3D = np.array([
            [-hat_w_3D/2,  hat_h_3D/2,  0],
            [ hat_w_3D/2,  hat_h_3D/2,  0],
            [ hat_w_3D/2, -hat_h_3D/2,  0],
            [-hat_w_3D/2, -hat_h_3D/2,  0]
        ], dtype=np.float32) + hat_center_3D

        hat_projected, _ = cv2.projectPoints(hat_3D, rvec, tvec, cam_matrix, dist_coeffs)
        hat_dst_pts = hat_projected.reshape(-1, 2).astype(np.float32)

        # ===== 모자 이미지 처리 =====
        hat_h, hat_w = hat_cropped_cache.shape[:2]
        new_hat_w = int(hat_w * z_scale)
        new_hat_h = int(hat_h * z_scale)
        
        hat_resized = cv2.resize(hat_cropped_cache, (new_hat_w, new_hat_h), 
                                interpolation=cv2.INTER_LINEAR)

        hat_src_pts = np.array([
            [0, 0],
            [new_hat_w - 1, 0],
            [new_hat_w - 1, new_hat_h - 1],
            [0, new_hat_h - 1]
        ], dtype=np.float32)

        H_hat = cv2.getPerspectiveTransform(hat_src_pts, hat_dst_pts)
        hat_warped = cv2.warpPerspective(hat_resized, H_hat, 
                                        (frame.shape[1], frame.shape[0]), 
                                        borderValue=(0, 0, 0, 0),
                                        flags=cv2.INTER_LINEAR)
        frame = overlay_transparent(frame, hat_warped, 0, 0)

        # ===== 산타 수염 =====
        nose_3D = model_points[0]
        chin_3D = model_points[1]
        beard_center_3D = (nose_3D + chin_3D) / 2 + np.array([0, -beard_h_3D * 0.2, eye_distance_3D * 0.2])
        
        beard_3D = np.array([
            [-beard_w_3D/2,  beard_h_3D/2,  0],
            [ beard_w_3D/2,  beard_h_3D/2,  0],
            [ beard_w_3D/2, -beard_h_3D/2,  0],
            [-beard_w_3D/2, -beard_h_3D/2,  0]
        ], dtype=np.float32) + beard_center_3D

        beard_projected, _ = cv2.projectPoints(beard_3D, rvec, tvec, cam_matrix, dist_coeffs)
        beard_dst_pts = beard_projected.reshape(-1, 2).astype(np.float32)

        # ===== 수염 이미지 처리 =====
        beard_h, beard_w = beard_cropped_cache.shape[:2]
        new_beard_w = int(beard_w * z_scale)
        new_beard_h = int(beard_h * z_scale)
        beard_resized = cv2.resize(beard_cropped_cache, (new_beard_w, new_beard_h), 
                                   interpolation=cv2.INTER_LINEAR)

        beard_src_pts = np.array([
            [0, 0],
            [new_beard_w - 1, 0],
            [new_beard_w - 1, new_beard_h - 1],
            [0, new_beard_h - 1]
        ], dtype=np.float32)

        H_beard = cv2.getPerspectiveTransform(beard_src_pts, beard_dst_pts)
        beard_warped = cv2.warpPerspective(beard_resized, H_beard, 
                                          (frame.shape[1], frame.shape[0]), 
                                          borderValue=(0, 0, 0, 0),
                                          flags=cv2.INTER_LINEAR)
        frame = overlay_transparent(frame, beard_warped, 0, 0)

    return frame


# ====== 입력 처리 ======
ext = os.path.splitext(input_path)[1].lower()

if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    result = apply_santa_filter_optimized(img)
    output_path = os.path.splitext(input_path)[0] + "_santa_fixed.png"
    cv2.imwrite(output_path, result)
    cv2.imshow("Santa Filter Fixed Size", result)
    print(f"✅ 이미지 결과 저장 완료: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_santa_fixed.mp4"
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    print("=" * 60)
    print("산타 필터 (첫 프레임 크기 고정)")
    print("ESC를 눌러 종료")
    print("=" * 60)
    
    frame_count = 0
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        result = apply_santa_filter_optimized(frame)
        out.write(result)
        cv2.imshow("Santa Filter Fixed Size", result)
        
        # FPS 표시
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"처리 중... {frame_count} 프레임 | FPS: {fps:.1f}")
        
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"\n✅ 영상 결과 저장 완료: {out_path}")
    print(f"   총 {frame_count} 프레임 처리됨")
