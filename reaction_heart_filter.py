import cv2
import dlib
import numpy as np
import os
import random

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = ["assets/heart1.png", "assets/heart2.png", "assets/heart3.png"]
input_path = "smile_girl3.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
filter_img = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in filter_path]

# 전역 변수
blink_counter = 0        # 눈 감은 연속 프레임 수
BLINK_REQUIRED = 4
heart_particles = [] 


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


# ===== 파티클 기반 하트 클래스 =====
class HeartParticle:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img.copy()
        self.size = np.random.randint(40, 80)     # 초기 크기
        self.blur = np.random.randint(20, 40)      # 블러 크기
        self.life = np.random.randint(10, 20)     # 지속 시간
        self.speed = np.random.uniform(-1.0, -2.0)  # 위로 이동 (음수)
        self.angle = np.random.uniform(-0.3, 0.3)   # 좌우 흔들림

        self.h0, self.w0 = self.img.shape[:2]

    def update(self):
        self.x += self.angle * 5
        self.y += self.speed              # 위 방향
        self.size += 1.0                  # 점점 커짐
        self.life -= 1

    def get_image(self):
        max_side = max(self.w0, self.h0)
        scale = self.size / max_side

        new_w = int(self.w0 * scale)
        new_h = int(self.h0 * scale)

        resized = cv2.resize(self.img, (new_w, new_h))
        return resized
    

# ===== 하트 생성 위치 계산 함수 =====
def get_outer_eye_position(landmarks, is_left=True):
    if is_left:
        eye_points = landmarks[36:42]   # 왼쪽 눈 6개 점
    else:
        eye_points = landmarks[42:48]   # 오른쪽 눈 6개 점

    eye_center = np.mean(eye_points, axis=0)
    face_center = np.mean(landmarks, axis=0)
    vec = face_center - eye_center
    vec = -vec

    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1
    direction = vec / norm

    dist = np.random.randint(100, 120)
    spawn_point = eye_center + direction * dist

    face_height = np.max(landmarks[:,1]) - np.min(landmarks[:,1])
    spawn_point[1] += int(face_height * 0.1) + np.random.randint(10, 25)

    return int(spawn_point[0]), int(spawn_point[1])


# ====== 메인 필터 함수 ======
def apply_heart_filter(frame):
    global blink_counter, heart_particles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        pts = np.array([[p.x, p.y] for p in predictor(gray, face).parts()])

        left_EAR = eye_aspect_ratio(pts[36:42])
        right_EAR = eye_aspect_ratio(pts[42:48])
        EAR = (left_EAR + right_EAR) / 2
        EAR_THRESHOLD = 0.18

        # 눈 감겼을 때
        if EAR < EAR_THRESHOLD:
            blink_counter += 1

            if blink_counter >= BLINK_REQUIRED:

                left_closed  = left_EAR  < EAR_THRESHOLD
                right_closed = right_EAR < EAR_THRESHOLD

                spawn_eyes = []
                if left_closed and not right_closed:
                    spawn_eyes = ["left"]
                elif right_closed and not left_closed:
                    spawn_eyes = ["right"]
                elif left_closed and right_closed:
                    spawn_eyes = ["left", "right"]

                # 각 눈에서 하트 생성
                for eye in spawn_eyes:
                    hx, hy = get_outer_eye_position(pts, is_left=(eye == "left"))

                    img = random.choice(filter_img)  # ← FIXED
                    heart_particles.append(
                        HeartParticle(hx, hy, img)
                    )

        else:
            blink_counter = 0  # ← 정상적으로 위치

    # 파티클 업데이트 + 렌더링
    new_particles = []
    for p in heart_particles:
        p.update()

        if p.life > 0:
            overlay = p.get_image()
            frame = overlay_transparent(
                frame,
                overlay,
                int(p.x - p.size / 2),
                int(p.y - p.size / 2)
            )
            new_particles.append(p)

    heart_particles = new_particles
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