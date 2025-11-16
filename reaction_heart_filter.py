import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets/heart.png"
input_path = "smile_girl3.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

# 전역 변수
blink_counter = 0        # 눈 감은 연속 프레임 수
BLINK_REQUIRED = 4
hearts = []          # 현재 하트 정보 (x, y, width, height, angle)
heart_timer = 0      # 하트 유지 프레임 카운터
HEART_DURATION = 10


# ===== EAR 계산 함수 =====
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# ===== 이미지 합성 =====
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


# ===== 블러된 PNG 만들기 =====
def create_blurred_filter(img, blur_size=15):
    if blur_size % 2 == 0:
        blur_size += 1

    b, g, r, a = cv2.split(img)
    rgb = cv2.merge([b, g, r])
    a_blur = cv2.GaussianBlur(a, (blur_size, blur_size), 0)
    blurred = cv2.merge([b, g, r, a_blur])

    return blurred

heart_filter = create_blurred_filter(filter_img, blur_size=25)

# ====== 메인 필터 함수 ======
def apply_heart_filter(frame):
    global blink_counter, hearts, heart_timer
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        pts = np.array([[p.x, p.y] for p in predictor(gray, face).parts()])
        left_EAR = eye_aspect_ratio(pts[36:42])
        right_EAR = eye_aspect_ratio(pts[42:48])
        EAR = (left_EAR + right_EAR) / 2
        EAR_THRESHOLD = 0.2

        # 얼굴 영역
        face_x1, face_y1 = face.left(), face.top()
        face_x2, face_y2 = face.right(), face.bottom()

        if EAR < EAR_THRESHOLD:
            blink_counter += 1
            # 눈을 충분히 감았으면 하트 생성
            if blink_counter >= BLINK_REQUIRED and heart_timer == 0:
                hearts = []
                for _ in range(np.random.randint(10, 13)):
                    w = np.random.randint(30, 100)
                    h = int(w * (heart_filter.shape[0] / heart_filter.shape[1]))

                    # 얼굴 영역 제외 랜덤 좌표
                    while True:
                        x = np.random.randint(0, frame.shape[1]-w)
                        y = np.random.randint(0, frame.shape[0]-h)
                        if x + w < face_x1 or x > face_x2 or y + h < face_y1 or y > face_y2:
                            break

                    angle = np.random.uniform(-30, 30)
                    blur_strength = np.random.randint(15, 40)
                    hearts.append((x, y, w, h, angle, blur_strength))

                heart_timer = HEART_DURATION
        else:
            blink_counter = 0

    if heart_timer > 0:
        for x, y, w, h, angle, blur_strength in hearts:
            resized = cv2.resize(heart_filter, (w, h))
            blurred_heart = create_blurred_filter(resized, blur_size=blur_strength)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(blurred_heart, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0,0,0,0))
            frame = overlay_transparent(frame, rotated, x, y)
        heart_timer -= 1  # 출력 후 감소

    return frame


# ====== 입력 처리 (영상 전용) ======
video_exts = [".mp4", ".avi", ".mov", ".mkv"]  # 영상 확장자 목록
ext = os.path.splitext(input_path)[1].lower()

if ext in video_exts:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_heart.mp4"
    out = cv2.VideoWriter(
        out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = apply_heart_filter(frame)
        out.write(result)
        cv2.imshow("AR Heart (Video)", result)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")