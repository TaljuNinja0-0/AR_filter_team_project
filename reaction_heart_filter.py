import cv2
import dlib
import numpy as np
import random
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = ["assets/heart.png"]
#input_path = "smile_girl3.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
filter_img = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in filter_path]

# ===== 전역 상태 =====
blink_counter = 0
eye_closed = False
heart_particles = []

BLINK_REQUIRED = 2
MAX_HEARTS = 60

EAR_OPEN  = 0.18   # 감김 진입
EAR_CLOSE = 0.22   # 열림 복귀

# ===== EAR 계산 =====
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ===== 알파 합성 =====
def overlay_transparent(bg, overlay, x, y):
    bh, bw = bg.shape[:2]
    h, w = overlay.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)

    ox1 = max(-x, 0)
    oy1 = max(-y, 0)
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return bg

    crop_bg = bg[y1:y2, x1:x2]
    crop_ov = overlay[oy1:oy2, ox1:ox2]

    if crop_ov.shape[2] < 4:
        return bg

    img = crop_ov[..., :3]
    alpha = crop_ov[..., 3:] / 255.0

    bg[y1:y2, x1:x2] = (1 - alpha) * crop_bg + alpha * img
    return bg.astype(np.uint8)

# ===== 하트 파티클 =====
class HeartParticle:
    def __init__(self, x, y, img, face_scale):
        self.x = x
        self.y = y
        self.img = img.copy()

        self.h0, self.w0 = self.img.shape[:2]

        self.size = face_scale * np.random.uniform(0.16, 0.26)
        self.grow = face_scale * np.random.uniform(0.0, 0.01)
        self.speed = -face_scale * np.random.uniform(0.01, 0.02)
        self.angle = np.random.uniform(-0.3, 0.3)

        self.alpha = np.random.uniform(0.5, 0.9)
        self.alpha_step = 0.8 / 30
        self.life = np.random.randint(18, 24)

    def update(self):
        self.x += self.angle * 3
        self.y += self.speed
        self.size += self.grow
        self.life -= 1
        self.alpha = max(0.0, self.alpha - self.alpha_step)

    def get_image(self):
        scale = self.size / max(self.w0, self.h0)
        new_w = int(self.w0 * scale)
        new_h = int(self.h0 * scale)

        resized = cv2.resize(self.img, (new_w, new_h))
        if resized.shape[2] == 4:
            resized = resized.copy()
            resized[:, :, 3] = (resized[:, :, 3] * self.alpha).astype(np.uint8)
        return resized

# ===== 눈 외측 생성 위치 =====
def get_outer_eye_position(landmarks, is_left=True):
    eye = landmarks[36:42] if is_left else landmarks[42:48]

    eye_center = np.mean(eye, axis=0)
    face_center = np.mean(landmarks, axis=0)

    direction = eye_center - face_center
    direction /= (np.linalg.norm(direction) + 1e-6)

    dist = np.random.randint(80, 120)
    spawn = eye_center + direction * dist

    face_h = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
    spawn[1] += face_h * 0.1 + np.random.randint(10, 50)

    return int(spawn[0]), int(spawn[1])

# ===== 메인 필터 =====
def apply_heart_filter(frame):
    global blink_counter, eye_closed, heart_particles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        pts = np.array([[p.x, p.y] for p in predictor(gray, face).parts()])

        left_EAR  = eye_aspect_ratio(pts[36:42])
        right_EAR = eye_aspect_ratio(pts[42:48])
        EAR = (left_EAR + right_EAR) / 2.0

        face_scale = (face.width() + face.height()) / 2.0
        emit_count = int(face_scale / 80)
        emit_count = np.clip(emit_count, 3, 6)

        # ===== 히스테리시스 =====
        if not eye_closed and EAR < EAR_OPEN:
            eye_closed = True
            blink_counter = 1

        elif eye_closed and EAR < EAR_OPEN:
            blink_counter += 1

            if blink_counter == BLINK_REQUIRED:
                spawn_eyes = []
                if left_EAR < EAR_OPEN and right_EAR >= EAR_OPEN:
                    spawn_eyes = ["left"]
                elif right_EAR < EAR_OPEN and left_EAR >= EAR_OPEN:
                    spawn_eyes = ["right"]
                elif left_EAR < EAR_OPEN and right_EAR < EAR_OPEN:
                    spawn_eyes = ["left", "right"]

                for eye in spawn_eyes:
                    for _ in range(emit_count):
                        if len(heart_particles) >= MAX_HEARTS:
                            break
                        x, y = get_outer_eye_position(pts, eye == "left")

                        img = random.choice(filter_img)
                        heart_particles.append(
                            HeartParticle(x, y, img, face_scale)
                        )

        elif eye_closed and EAR > EAR_CLOSE:
            eye_closed = False
            blink_counter = 0

    # ===== 파티클 업데이트 =====
    new_particles = []
    for p in heart_particles:
        p.update()
        if p.life > 0:
            overlay = p.get_image()
            frame = overlay_transparent(
                frame,
                overlay,
                int(p.x - overlay.shape[1] / 2),
                int(p.y - overlay.shape[0] / 2)
            )
            new_particles.append(p)

    heart_particles = new_particles
    return frame

'''
# ===== 영상 처리 =====
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = os.path.splitext(input_path)[0] + "_heart.mp4"

out = cv2.VideoWriter(
    out_path,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = apply_heart_filter(frame)
    out.write(result)
    cv2.imshow("AR Heart", result)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 영상 결과 저장 완료: {out_path}")
'''