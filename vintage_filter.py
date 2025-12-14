import cv2
import numpy as np

filter_path = "assets_moving/vintage_1.mp4"
cap_filter = cv2.VideoCapture(filter_path)

_cached_overlay = None       # resize된 필터 프레임
_cached_alpha = None         # 그린스크린 alpha
_cached_noise = None         # 노이즈 캐시
_last_frame = None           # 프레임 드랍용
_frame_count = 0             # 내부 프레임 카운트


def overlay_transparent(background, overlay, alpha_mask):
    alpha = (alpha_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = overlay.astype(np.float32) * alpha + \
              background.astype(np.float32) * (1.0 - alpha)
    return blended.astype(np.uint8)


def remove_green_background(frame):
    b, g, r = cv2.split(frame)
    diff = g.astype(np.int16) - np.maximum(r, b).astype(np.int16)

    diff_thresh = 30
    g_min = 60

    alpha = np.clip(
        (diff_thresh - diff) * (255.0 / diff_thresh),
        0, 255
    ).astype(np.uint8)

    strong_green = (diff > diff_thresh + 30) & (g > g_min + 40)
    alpha[strong_green] = 0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    alpha[hsv[:, :, 2] < 70] = 255

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    return alpha

def apply_vintage_filter(frame):
    global _cached_overlay, _cached_alpha
    global _cached_noise, _last_frame, _frame_count

    h, w = frame.shape[:2]
    _frame_count += 1

    # ---------- 필터 영상 캐싱 ----------
    if _cached_overlay is None or _cached_overlay.shape[:2] != (h, w):
        ret, filter_frame = cap_filter.read()
        if not ret:
            cap_filter.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, filter_frame = cap_filter.read()
            if not ret:
                return frame

        _cached_overlay = cv2.resize(filter_frame, (w, h))
        _cached_alpha = remove_green_background(_cached_overlay)

    # ---------- 오버레이 ----------
    result = overlay_transparent(frame, _cached_overlay, _cached_alpha)

    # ---------- 노이즈 (5프레임마다 갱신) ----------
    if _cached_noise is None or _frame_count % 5 == 0:
        _cached_noise = np.random.normal(0, 12, frame.shape).astype(np.int16)

    result = np.clip(
        result.astype(np.int16) + _cached_noise,
        0, 255
    ).astype(np.uint8)

    # ----------색감/대비 감소 ----------
    faded = cv2.convertScaleAbs(result, alpha=0.85, beta=-10)
    faded = cv2.cvtColor(faded, cv2.COLOR_BGR2GRAY)
    faded = cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.65, faded, 0.35, 0)

    # ---------- 컬러 시프트 ----------
    b, g, r = cv2.split(result)
    r = cv2.add(r, 20)
    b = cv2.subtract(b, 15)
    result = cv2.merge([b, g, r])

    # ---------- HSV 톤 조정 ----------
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + 6) % 180
    hsv[..., 1] *= 0.9
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.05, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ---------- 블러 ----------
    result = cv2.GaussianBlur(result, (3, 3), 0.6)

    # ---------- 프레임 드랍 (FPS 제한) ----------
    if _frame_count % 3 != 0 and _last_frame is not None:
        return _last_frame

    _last_frame = result.copy()
    return result



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
'''