import cv2
import dlib
import numpy as np
import os


# ====== 설정 ======
predictor_path = "shape_predictor_68_face_landmarks.dat"
filter_path = "assets_moving/vintage_1.mp4"
input_path = "smile_girl.mp4" #"Lena.jpg"

ext = os.path.splitext(input_path)[1].lower()
cap_filter = cv2.VideoCapture(filter_path)


# ==== 오버레이 ====
def overlay_transparent(background, overlay, alpha_mask): # 원본 프레임(BRG), 필터 프레임(BRG), 그린스크린
    overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
    alpha_mask = cv2.resize(alpha_mask, (background.shape[1], background.shape[0]))

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
    # 이미지/조명에 따라 튜닝
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

    # 가우시안 노이즈 추가
    noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 대비, 색감 줄이기
    faded = cv2.convertScaleAbs(result, alpha=0.85, beta=-10)
    faded = cv2.cvtColor(faded, cv2.COLOR_BGR2GRAY)
    faded = cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.65, faded, 0.35, 0)

    # 컬러 시프트 ( RGB 약간씩 밀기)
    b, g, r = cv2.split(result)
    r = cv2.add(r, 25)   # 따뜻한 톤 강조
    b = cv2.subtract(b, 20)  # 푸른기 줄임
    result = cv2.merge([b, g, r])
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 색조 변환 (Hue 살짝 이동)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + 8) % 180
    hsv[..., 1] = hsv[..., 1] * 0.9         
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.05, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 가우시안 블러
    result = cv2.GaussianBlur(result, (3, 3), 0.7)

    # 프레임 드랍
    if frame_idx % 3 != 0:  # 5프레임 중 4프레임은 이전 프레임 유지
        return apply_vintage_filter.last_frame
    apply_vintage_filter.last_frame = result.copy()

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
        cv2.imshow("Vintage Filter", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("output_vintage.png", result)
    filter_cap.release()

else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_vintage.mp4"
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_vintage_filter(frame, cap_filter, frame_idx)
        out.write(result)
        cv2.imshow("Vintage Filter", result)
        if cv2.waitKey(1) == 27:
            break
        frame_idx += 1

    cap.release()
    cap_filter.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 영상 결과 저장 완료: {out_path}")