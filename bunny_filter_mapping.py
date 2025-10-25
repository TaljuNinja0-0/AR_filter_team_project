import cv2
import dlib
import numpy as np
import os

# === 초기 설정 ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === 필터 이미지 ===
filter_img = cv2.imread("bunny_filter.png", cv2.IMREAD_UNCHANGED)

def overlay_transparent(background, overlay, x, y):
    """알파 채널 기반 투명 합성 (좌표 경계 체크 포함)"""
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    # 오버레이가 화면을 완전히 벗어난 경우
    if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
        return background

    # 잘려야 하는 부분 계산
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, bw)
    y2 = min(y + h, bh)

    overlay_x1 = max(-x, 0)
    overlay_y1 = max(-y, 0)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # 슬라이싱
    background_roi = background[y1:y2, x1:x2]
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return background

    if overlay_crop.shape[2] < 4:
        return background

    overlay_img = overlay_crop[..., :3]
    mask = overlay_crop[..., 3:] / 255.0

    # 알파 블렌딩
    blended = (1 - mask) * background_roi + mask * overlay_img
    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background



def apply_filter(frame):
    """단일 프레임(이미지)에 필터 적용"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # 주요 포인트
        nose = points[30]
        left_eye = points[36:42]
        right_eye = points[42:48]

        # 얼굴 회전
        left_eye_center = left_eye.mean(axis=0).astype(int)
        right_eye_center = right_eye.mean(axis=0).astype(int)

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # 얼굴 크기
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()

        # 필터 크기 조정
        filter_scaled = cv2.resize(filter_img, (int(face_width * 1.5), int(face_height * 1.5)))

        # 회전
        center = (filter_scaled.shape[1] // 2, filter_scaled.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_filter = cv2.warpAffine(
            filter_scaled,
            rot_mat,
            (filter_scaled.shape[1], filter_scaled.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # 위치 조정
        x = int(nose[0] - rotated_filter.shape[1] / 2)
        y = int(nose[1] - rotated_filter.shape[0] * 0.7)

        # 합성
        frame = overlay_transparent(frame, rotated_filter, x, y)

    return frame


# === 입력 파일 설정 ===
input_path = "smile_girl.mp4"
output_path = "output_with_filter.mp4"

if os.path.splitext(input_path)[1].lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
    # 이미지 입력
    img = cv2.imread(input_path)
    result = apply_filter(img)
    cv2.imwrite("output_image.png", result)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # 영상 입력
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_filter(frame)
        out.write(result)
        cv2.imshow("Processing...", result)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("필터 적용 완료!")
