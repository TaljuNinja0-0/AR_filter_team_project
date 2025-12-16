import cv2
import dlib
import numpy as np
import os
from collections import deque

# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
input_path = "smile_girl2.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ===== 전역 변수 =====
MAX_TRAIL = 8       # 유지할 프레임 수
ALPHA = 0.15          # 잔상 투명도
face_trail = deque()

# ===== 얼굴 영역 추출 =====
def get_face_mask(frame, face):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    hull = cv2.convexHull(points)

    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    face_img = cv2.bitwise_and(frame, mask)
    return face_img

# ===== 메인 필터 =====
def apply_ghost_filter(frame):
    global face_trail

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    face_layer = np.zeros_like(frame)
    for face in faces:
        face_img = get_face_mask(frame, face)
        face_layer = cv2.add(face_layer, face_img)

    # 큐 업데이트
    face_trail.append(face_layer)
    if len(face_trail) > MAX_TRAIL:
        face_trail.popleft()

    # 트레일 합성
    output = frame.copy()
    for i, past_face in enumerate(face_trail):
        alpha = ALPHA * (1 - i / MAX_TRAIL)  # 오래된 얼굴일수록 더 투명
        mask = past_face.astype(bool)        # 얼굴 영역만 오버레이
        output[mask] = cv2.addWeighted(output, 1 - alpha, past_face, alpha, 0)[mask]

    return output

# ===== 영상 처리 =====
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = os.path.splitext(input_path)[0] + "_ghost_trail.mp4"

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

    result = apply_ghost_filter(frame)
    out.write(result)
    cv2.imshow("Ghost Trail", result)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 영상 결과 저장 완료: {out_path}")
