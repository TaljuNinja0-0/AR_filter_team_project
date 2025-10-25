import cv2
import dlib
import numpy as np

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets/glasses_filter.png"
input_path = "smile_girl.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
orb = cv2.ORB_create(nfeatures=500)

filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

# 안경 이미지에서 브릿지(코에 걸치는 중앙) 좌표
# PNG 기준으로 브릿지 위치를 직접 지정 (실제 안경 이미지에 따라 조정 필요)
bridge_x = filter_img.shape[1] // 2
bridge_y = int(filter_img.shape[0] * 0.45)  # 안경 이미지 상단에서 브릿지 높이 비율

def overlay_transparent(background, overlay, x, y):
    """알파 채널 기반 투명 합성"""
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

def curve_warp_glasses(img, left_curvature=0.2, right_curvature=0.2):
    """
    양쪽 렌즈 기준으로 곡률을 다르게 주는 입체 안경 변형
    """
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    center_x = w // 2

    for y in range(h):
        for x in range(w):
            # 왼쪽 렌즈 (좌측 절반)
            if x < center_x:
                offset = left_curvature * ((x - center_x) ** 2) / (center_x ** 2)
            else:  # 오른쪽 렌즈 (우측 절반)
                offset = right_curvature * ((x - center_x) ** 2) / (center_x ** 2)
            map_x[y, x] = x
            map_y[y, x] = y + offset * h * 0.4  # y방향 왜곡
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return warped


def apply_glasses_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        for idx, (x, y) in enumerate(points):
            # 각 점에 원(파란색) 표시
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            # 각 점 옆에 번호(노란색 글씨) 표시
            cv2.putText(frame, str(idx), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # 눈 좌표
        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        eye_center = ((left_eye + right_eye) / 2).astype(int)

        # 눈 각도 계산
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                      right_eye[0] - left_eye[0]))

        # 스케일 계산
        eye_width = np.linalg.norm(right_eye - left_eye)
        scale = (eye_width / filter_img.shape[1]) * 3.0

        # 안경 크기 조정
        new_w = int(filter_img.shape[1] * scale)
        new_h = int(filter_img.shape[0] * scale)
        resized_filter = cv2.resize(filter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 회전 적용
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1)
        rotated_filter = cv2.warpAffine(resized_filter, M, (new_w, new_h), borderValue=(0, 0, 0, 0))

        curved_filter = curve_warp_glasses(rotated_filter, left_curvature=0.20, right_curvature=0.20)

        face_center = points[27]

        x = int(face_center[0] - curved_filter.shape[1] / 2)
        y = int(face_center[1] - curved_filter.shape[0] / 2.5)

        frame = overlay_transparent(frame, curved_filter, x, y)

    return frame


# ====== 이미지/영상 처리 ======
if input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
    img = cv2.imread(input_path)
    result = apply_glasses_filter(img)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_glasses.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_glasses_filter(frame)
        out.write(result)
        cv2.imshow("Processing...", result)
        if cv2.waitKey(1) == 27:  # ESC 키로 종료
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
