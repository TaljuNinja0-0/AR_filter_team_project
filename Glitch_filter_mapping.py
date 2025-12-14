import cv2
import dlib
import numpy as np
import os
import random

predictor_path = "shape_predictor_68_face_landmarks.dat"
#input_path = "tiny_girl.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def shape_to_np(shape):
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)


def expand_mask(mask, expansion_pixels):

    kernel = np.ones((expansion_pixels, expansion_pixels), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    return expanded_mask


def glitch_effect_face_chaos(frame, landmarks):

    h, w = frame.shape[:2]
    
    # 설정
    max_offset = 55
    num_blocks = 18
    expansion = 90
    
    # 원본 얼굴 마스크
    original_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(original_mask, hull, 255)
    
    # 마스크 확장
    expanded_mask = expand_mask(original_mask, expansion)
    
    # 경계 상자
    x, y, w_face, h_face = cv2.boundingRect(landmarks)
    padding = expansion
    x = max(0, x - padding)
    y = max(0, y - padding)
    w_face = min(w - x, w_face + padding * 2)
    h_face = min(h - y, h_face + padding * 2)
    
    result = frame.copy()
    block_height = max(h_face // num_blocks, 5)
    
    # 블록별 처리
    for i in range(0, h_face, block_height):
        y_start = y + i
        y_end = min(y + i + block_height, y + h_face)
        
        if y_end >= h:
            break
        
        block = frame[y_start:y_end, x:x+w_face].copy()
        mask_block = expanded_mask[y_start:y_end, x:x+w_face]
        
        # RGB 분리
        b, g, r = cv2.split(block)
        
        # 완전 랜덤 이동
        r_x = random.randint(-max_offset, max_offset)
        r_y = random.randint(-max_offset//2, max_offset//2)
        M_r = np.float32([[1, 0, r_x], [0, 1, r_y]])
        r_shifted = cv2.warpAffine(r, M_r, (w_face, y_end - y_start))
        mask_r = cv2.warpAffine(mask_block, M_r, (w_face, y_end - y_start))
        
        g_x = random.randint(-max_offset, max_offset)
        g_y = random.randint(-max_offset//2, max_offset//2)
        M_g = np.float32([[1, 0, g_x], [0, 1, g_y]])
        g_shifted = cv2.warpAffine(g, M_g, (w_face, y_end - y_start))
        mask_g = cv2.warpAffine(mask_block, M_g, (w_face, y_end - y_start))
        
        b_x = random.randint(-max_offset, max_offset)
        b_y = random.randint(-max_offset//2, max_offset//2)
        M_b = np.float32([[1, 0, b_x], [0, 1, b_y]])
        b_shifted = cv2.warpAffine(b, M_b, (w_face, y_end - y_start))
        mask_b = cv2.warpAffine(mask_block, M_b, (w_face, y_end - y_start))
        
        # 채널별 합성
        r_mask_norm = (mask_r / 255.0).astype(np.float32)
        g_mask_norm = (mask_g / 255.0).astype(np.float32)
        b_mask_norm = (mask_b / 255.0).astype(np.float32)
        
        result[y_start:y_end, x:x+w_face, 2] = (
            r_mask_norm * r_shifted + 
            (1 - r_mask_norm) * result[y_start:y_end, x:x+w_face, 2]
        ).astype(np.uint8)
        
        result[y_start:y_end, x:x+w_face, 1] = (
            g_mask_norm * g_shifted + 
            (1 - g_mask_norm) * result[y_start:y_end, x:x+w_face, 1]
        ).astype(np.uint8)
        
        result[y_start:y_end, x:x+w_face, 0] = (
            b_mask_norm * b_shifted + 
            (1 - b_mask_norm) * result[y_start:y_end, x:x+w_face, 0]
        ).astype(np.uint8)
    
    return result


def apply_filter(frame):
    # 영상 어둡게 (50% 밝기)
    frame = (frame * 0.5).astype(np.uint8)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if len(faces) == 0:
        return frame
    
    for face in faces:
        try:
            shape = predictor(gray, face)
            landmarks = shape_to_np(shape)
            frame = glitch_effect_face_chaos(frame, landmarks)
            
        except Exception as e:
            print(f"얼굴 처리 중 오류 발생: {e}")
            continue
    
    return frame

'''
# ====== 입력 처리 ======
ext = os.path.splitext(str(input_path))[1].lower()

if ext in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(input_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {input_path}")
    else:
        print("얼굴 형태 글리치 필터 적용 중...")
        result = apply_filter(img)
        
        output_path = os.path.splitext(input_path)[0] + "_glitch.png"
        cv2.imwrite(output_path, result)
        print(f"글리치 이미지 저장 완료: {output_path}")
        
        cv2.imshow("Face-Shaped Glitch Filter", result)
        print("아무 키나 누르면 종료됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"영상을 불러올 수 없습니다: {input_path}")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.splitext(input_path)[0] + "_glitch.mp4"
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        print(f"영상 처리 시작... (총 {total_frames} 프레임)")
        
        frame_count = 0
        show_display = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = apply_filter(frame)
            out.write(result)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"처리 중... {frame_count}/{total_frames} 프레임 ({progress:.1f}%)")
            
            if show_display:
                display_frame = cv2.resize(result, (width//2, height//2))
                cv2.imshow("Face Glitch Filter", display_frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    show_display = False
                    cv2.destroyAllWindows()
                    print("\n창 표시를 중단했습니다. 백그라운드에서 계속 처리 중...\n")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n영상 결과 저장 완료: {out_path}")
  
'''