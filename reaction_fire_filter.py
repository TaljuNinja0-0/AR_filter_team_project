import cv2
import dlib
import numpy as np
import os

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
fire_path = "assets/fire.png"
input_path = "mad_girl.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fire_img = cv2.imread(fire_path, cv2.IMREAD_UNCHANGED)

# ===== 전역 상태 =====
fire_particles = []
mouth_open = False

# ===== MAR 계산 =====
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    D = np.linalg.norm(mouth[12] - mouth[16])
    return (A + B + C) / (3.0 * D)

def preprocess_alpha_edge(img, ksize=7):
    if img is None or img.shape[2] < 4:
        return img

    img = img.copy()
    alpha = img[..., 3]
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    img[..., 3] = alpha
    return img

fire_img = preprocess_alpha_edge(fire_img, ksize=17)

# ===== 알파 합성 =====
def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
        return background

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)
    ox1 = max(-x, 0)
    oy1 = max(-y, 0)
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    bg_roi = background[y1:y2, x1:x2]
    overlay_crop = overlay[oy1:oy2, ox1:ox2]

    if overlay_crop.shape[2] < 4:
        return background

    overlay_img = overlay_crop[..., :3]
    mask = overlay_crop[..., 3:] / 255.0
    blended = (1 - mask) * bg_roi + mask * overlay_img
    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background

# ===== 파티클 클래스 =====
class FireParticle:
    def __init__(self, x, y, img, base_scale):
        self.x = x
        self.y = y
        self.img = img.copy()
        self.size = int(base_scale * np.random.uniform(0.4, 0.6))
        self.life = np.random.randint(24, 30)
        self.speed = np.random.uniform(32.0, 36.0)
        self.angle = np.random.uniform(-0.2, 0.2)

    def update(self):
        self.x += self.angle * 5
        self.y += self.speed
        self.size += 24.0
        self.life -= 1

    def get_image(self):
        return cv2.resize(self.img, (int(self.size), int(self.size)))

# ===== 메인 필터 =====
def apply_fire_filter(frame):
    global fire_particles, mouth_open

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    OPEN = 0.75
    CLOSE = 0.55

    for face in faces:
        pts = np.array([[p.x, p.y] for p in predictor(gray, face).parts()])
        mouth = pts[48:68]
        MAR = mouth_aspect_ratio(mouth)

        # 얼굴 크기 기반 스케일 계산
        face_size = (face.width() + face.height()) / 2
        fire_scale = face_size * 0.5

        # ===== 입 상태 머신 =====
        if not mouth_open and MAR > OPEN:
            mouth_open = True
            mouth_center = np.mean(mouth, axis=0).astype(int)

            for _ in range(np.random.randint(1, 2)):
                fire_particles.append(
                    FireParticle(
                        mouth_center[0],
                        mouth_center[1],
                        fire_img,
                        fire_scale
                    )
                )

        elif mouth_open and MAR < CLOSE:
            mouth_open = False

    # ===== 파티클 업데이트 =====
    new_particles = []
    for p in fire_particles:
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

    fire_particles = new_particles
    return frame

# ===== 영상 처리 =====
video_exts = [".mp4", ".avi", ".mov", ".mkv"]
ext = os.path.splitext(input_path)[1].lower()

if ext in video_exts:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_fire.mp4"

    out = cv2.VideoWriter(
        out_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_fire_filter(frame)
        out.write(result)

        cv2.imshow("AR Fire (Video)", result)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")
