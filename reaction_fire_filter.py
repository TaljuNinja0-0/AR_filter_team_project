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

# 전역 변수
fire_particles = []

# ===== MAR 계산 =====
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    D = np.linalg.norm(mouth[12] - mouth[16])
    MAR = (A + B + C) / (3.0 * D)
    return MAR

# ===== 알파 기반 합성 =====
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

# ===== 파티클 클래스 =====
class FireParticle:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img.copy()
        self.size = np.random.randint(80, 120)
        self.life = np.random.randint(8, 12)
        self.speed = np.random.uniform(8.0, 12.0)
        self.angle = np.random.uniform(-0.2, 0.2)

    def update(self):
        self.x += self.angle * 5
        self.y += self.speed  # 아래 방향
        self.size += 4.0  # 점점 커지도록
        self.life -= 1

    def get_image(self):
        resized = cv2.resize(self.img, (int(self.size), int(self.size)))
        return resized
    

# ====== 메인 필터 함수 ======
def apply_fire_filter(frame):
    global fire_particles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        pts = np.array([[p.x, p.y] for p in predictor(gray, face).parts()])
        mouth = pts[48:68]
        MAR = mouth_aspect_ratio(mouth)
        MAR_THRESHOLD = 0.6

        if MAR > MAR_THRESHOLD:
            mouth_center = np.mean(mouth, axis=0).astype(int)
            # 입에서 2~3개의 파티클만 생성
            for _ in range(np.random.randint(2, 4)):
                fire_particles.append(FireParticle(mouth_center[0], mouth_center[1], fire_img))

    # 파티클 업데이트 및 합성
    new_particles = []
    for p in fire_particles:
        p.update()
        if p.life > 0:
            overlay = p.get_image()
            frame = overlay_transparent(frame, overlay, int(p.x - p.size/2), int(p.y - p.size/2))
            new_particles.append(p)
    fire_particles = new_particles

    return frame

# ====== 입력 처리 (영상 전용) ======
video_exts = [".mp4", ".avi", ".mov", ".mkv"]
ext = os.path.splitext(input_path)[1].lower()

if ext in video_exts:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_fire.mp4"
    out = cv2.VideoWriter(
        out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = apply_fire_filter(frame)
        out.write(result)
        cv2.imshow("AR Fire (Video)", result)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")
