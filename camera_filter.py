import cv2
import numpy as np


# ====== 설정 ======
filter_path = "assets_moving/camera.mp4"
#input_path = "smile_girl.mp4" #"Lena.jpg"

#ext = os.path.splitext(input_path)[1].lower()
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

# 채도 감소 (녹색 > 회색)
def desaturate_green_ui(frame, alpha_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # 배경이 아닌 영역만
    ui_mask = (alpha_mask > 10)

    green_ui = (h > 35) & (h < 85) & ui_mask

    s[green_ui] *= 0.15  # 회색화 강도

    hsv = cv2.merge([
        h,
        np.clip(s, 0, 255),
        v
    ])

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)



# ====== 그린스크린 제거 함수 ======
def remove_green_background(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # 네온 그린 배경만 잡기
    green_bg = (
        (h > 35) & (h < 85) &     # green hue
        (s > 120) &              # 채도가 높은 초록만
        (v > 150)                # 밝은 초록만
    )

    alpha = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    alpha[green_bg] = 0

    # 가장자리 부드럽게
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    return alpha


# 필터 적용 함수
def apply_camera_filter(frame):
    global cap_filter

    if not cap_filter.isOpened():
        cap_filter.open(filter_path)

    ret_f, filter_frame = cap_filter.read()
    if not ret_f:
        cap_filter.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_f, filter_frame = cap_filter.read()
        if not ret_f:
            return frame

    alpha_mask = remove_green_background(filter_frame)
    filter_frame = desaturate_green_ui(filter_frame, alpha_mask)
    return overlay_transparent(frame, filter_frame, alpha_mask)



'''
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

'''