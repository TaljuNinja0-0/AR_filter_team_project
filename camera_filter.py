import cv2
import dlib
import numpy as np
import os


# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets_moving/camera.mp4"
input_path = "smile_girl.mp4" #"Lena.jpg"

ext = os.path.splitext(input_path)[1].lower()
cap_filter = cv2.VideoCapture(filter_path)


# ==== 오버레이 ====
def overlay_transparent(background, overlay, alpha_mask):
    # background, overlay: BGR uint8, alpha_mask: single-channel uint8(0~255)
    overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
    alpha_mask = cv2.resize(alpha_mask, (background.shape[1], background.shape[0]))

    # 0~1 범위로 정규화된 float alpha (3채널로 확장)
    alpha = (alpha_mask.astype(np.float32) / 255.0)[:, :, None]
    overlay_f = overlay.astype(np.float32)
    background_f = background.astype(np.float32)

    # 알파가 1일수록 overlay가 보이고 0이면 background 유지
    blended = overlay_f * alpha + background_f * (1.0 - alpha)
    return blended.astype(np.uint8)

# ====== 그린스크린 제거 함수 ======
def remove_green_background(frame):
    b, g, r = cv2.split(frame)
    diff = (g.astype(np.int16) - np.maximum(r, b).astype(np.int16)).astype(np.int16)

    # diff가 작으면 전경(불투명)으로 처리
    # 이미지/조명에 따라 튜닝하세요
    diff_thresh = 30   # diff가 이 값보다 작으면 전경
    g_min = 60         # 녹색으로 판단하려면 G가 최소 이 값 이상

    # diff가 클수록 투명(0)에 가깝게, 작을수록 불투명(255)
    alpha = np.clip((diff_thresh - diff) * (255.0 / max(1, diff_thresh)), 0, 255).astype(np.uint8)


    mask_strong_green = (diff > (diff_thresh + 30)) & (g > (g_min + 40))
    alpha[mask_strong_green] = 0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    alpha[v < 70] = 255

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    return alpha

# 필터 적용 함수
def apply_vintage_filter(frame, filter_cap, frame_idx):
    ret_f, filter_frame = filter_cap.read()
    if not ret_f:
        filter_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_f, filter_frame = filter_cap.read()

    # 초록색 제거 마스크 생성
    alpha_mask = remove_green_background(filter_frame)
    result = overlay_transparent(frame, filter_frame, alpha_mask)

    # 오버레이 합성
    return result

# 필터 적용
# 이미지 입력일 경우 (테스트용)
if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    filter_cap = cv2.VideoCapture(filter_path)
    ret, filter_frame = filter_cap.read()
    if ret:
        alpha_mask = remove_green_background(filter_frame)
        result = overlay_transparent(img, filter_frame, alpha_mask)
        cv2.imshow("Camera Filter", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("output_Camera.png", result)
    filter_cap.release()

else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_Camera_Filter.mp4"
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_vintage_filter(frame, cap_filter, frame_idx)
        out.write(result)
        cv2.imshow("Camera Filter", result)
        if cv2.waitKey(1) == 27:
            break
        frame_idx += 1

    cap.release()
    cap_filter.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")