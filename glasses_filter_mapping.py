import cv2
import dlib
import numpy as np

# ====== 사전 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "glasses_filter.png"
input_path = "smile_girl.mp4"  # or "video.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
sift = cv2.SIFT_create()

# ====== 필터 이미지 로드 ======
filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

def overlay_transparent(background, overlay, x, y):
    """알파 채널을 이용한 투명 합성"""
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # 왼쪽/오른쪽 눈 중심 계산
        left_eye_center = points[36:42].mean(axis=0)
        right_eye_center = points[42:48].mean(axis=0)
        nose_center = points[33]

        # 회전 각도 계산
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # ROI 추출 (눈 주변)
        x_min = int(min(left_eye_center[0], right_eye_center[0]) - 50)
        x_max = int(max(left_eye_center[0], right_eye_center[0]) + 50)
        y_min = int(min(left_eye_center[1], right_eye_center[1]) - 60)
        y_max = int(max(left_eye_center[1], right_eye_center[1]) + 80)

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, frame.shape[1])
        y_max = min(y_max, frame.shape[0])

        roi = gray[y_min:y_max, x_min:x_max]

        # SIFT 특징점 매칭 (필터 vs ROI)
        kp1, des1 = sift.detectAndCompute(filter_img[..., :3], None)
        kp2, des2 = sift.detectAndCompute(roi, None)

        if des1 is None or des2 is None:
            continue

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 4:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Homography 적용
        warped = cv2.warpPerspective(filter_img, H, (roi.shape[1], roi.shape[0]),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # 회전 정렬 및 위치 조정
        filter_center = (x_min, y_min)
        frame = overlay_transparent(frame, warped, x_min, y_min)

    return frame


# ====== 이미지 or 영상 모드 ======
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
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
