import cv2
import dlib
import numpy as np
import os

predictor_path = "shape_predictor_68_face_landmarks.dat"
#input_path = "mad_girl.mp4"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 3D 모델 포인트
model_points = np.array([
    (0.0, 0.0, 0.0),           # 코 끝
    (0.0, -63.6, -12.0),       # 턱
    (-43.3, 32.7, -26.0),      # 왼쪽 눈 바깥
    (43.3, 32.7, -26.0),       # 오른쪽 눈 바깥
    (-28.9, -28.9, -24.1),     # 왼쪽 입
    (28.9, -28.9, -24.1)       # 오른쪽 입
], dtype=np.float32)

def shape_to_np(shape):
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)

def warp_and_overlay(frame, region_points, scale=1.5, padding=20):
    # ROI 사각형 + padding
    x, y, w, h = cv2.boundingRect(region_points)
    x = max(0, x - padding); y = max(0, y - padding)
    w = min(frame.shape[1]-x, w + 2*padding)
    h = min(frame.shape[0]-y, h + 2*padding)

    roi = frame[y:y+h, x:x+w]

    # 기본 마스크 (눈/입 영역)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = region_points - [x, y]
    cv2.fillConvexPoly(mask, cv2.convexHull(shifted), 255)

    # 확대
    roi_big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    mask_big = cv2.resize(mask, (roi_big.shape[1], roi_big.shape[0]))
    kernel = np.ones((15,15), np.uint8)
    mask_dilated = cv2.dilate(mask_big, kernel, iterations=1)
    mask_feather = cv2.GaussianBlur(mask_dilated, (31,31), 15)
    mask_combined = np.where(mask_big==255, 255, mask_feather).astype(np.uint8)

    # 호모그래피
    src_pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    dst_pts = np.array([[0,0],[roi_big.shape[1],0],
                        [roi_big.shape[1],roi_big.shape[0]],[0,roi_big.shape[0]]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(roi, H, (roi_big.shape[1], roi_big.shape[0]))

    # 합성 위치
    cx, cy = x + w//2, y + h//2
    x1, y1 = cx - roi_big.shape[1]//2, cy - roi_big.shape[0]//2
    x2, y2 = x1 + roi_big.shape[1], y1 + roi_big.shape[0]

    # 클리핑
    x1c, y1c = max(0,x1), max(0,y1)
    x2c, y2c = min(frame.shape[1],x2), min(frame.shape[0],y2)
    roi_cropped = warped[(y1c-y1):(y2c-y1),(x1c-x1):(x2c-x1)]
    mask_cropped = mask_combined[(y1c-y1):(y2c-y1),(x1c-x1):(x2c-x1)]

    # 알파 블렌딩
    background = frame[y1c:y2c, x1c:x2c]
    alpha = mask_cropped.astype(float)/255.0
    blended = (1-alpha)[:,:,None]*background + alpha[:,:,None]*roi_cropped
    frame[y1c:y2c, x1c:x2c] = blended.astype(np.uint8)
    return frame


def apply_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    # 얼굴이 없으면 원본 반환
    if len(faces) == 0:
        return frame

    # 모든 얼굴에 대해 필터 적용
    for face in faces:
        try:
            # 각 얼굴의 랜드마크 검출
            shape = predictor(gray, face)
            landmarks = shape_to_np(shape)

            # PnP (카메라 포즈 추정)
            image_points = np.array([
                landmarks[30], landmarks[8],
                landmarks[36], landmarks[45],
                landmarks[48], landmarks[54]
            ], dtype=np.float32)
            
            h, w = frame.shape[:2]
            cam_matrix = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float32)
            dist_coeffs = np.zeros((4,1))
            cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

            # 눈/입 영역 확대 (왼쪽 눈, 오른쪽 눈, 입)
            frame = warp_and_overlay(frame, landmarks[36:42], scale=1.8, padding=25)
            frame = warp_and_overlay(frame, landmarks[42:48], scale=1.8, padding=25)
            frame = warp_and_overlay(frame, landmarks[48:60], scale=1.5, padding=30)
            
        except Exception as e:
            # 특정 얼굴에서 오류가 발생해도 다른 얼굴은 계속 처리
            print(f"얼굴 처리 중 오류 발생: {e}")
            continue
    
    return frame


# ====== 입력 처리 ======
ext = os.path.splitext(str(input_path))[1].lower()

if ext in [".jpg",".jpeg",".png"]:
    # 사진 처리
    img = cv2.imread(input_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {input_path}")
    else:
        print("필터 적용 중...")
        result = apply_filter(img)
        
        # 결과 저장 (추가된 부분)
        output_path = os.path.splitext(input_path)[0] + "_funny.png"
        cv2.imwrite(output_path, result)
        print(f"필터 적용된 이미지 저장 완료: {output_path}")
        
        # 결과 표시
        cv2.imshow("Pose-stable Funny Filter (Image)", result)
        print("아무 키나 누르면 종료됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    # 영상 처리
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"영상을 불러올 수 없습니다: {input_path}")
    else:
        # 영상 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.splitext(input_path)[0] + "_funny.mp4"
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        print(f"영상 처리 시작... (총 {total_frames} 프레임)")
        print("처리 중에는 창이 느릴 수 있지만, 저장된 영상은 정상 속도입니다.")
        print("빠르게 처리하려면 ESC를 눌러 창 표시를 건너뛸 수 있습니다.\n")
        
        frame_count = 0
        show_display = True  # 창 표시 여부
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 필터 적용
            result = apply_filter(frame)
            
            # 결과 저장 (항상 저장)
            out.write(result)
            
            frame_count += 1
            
            # 10프레임마다 진행 상황 출력
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"처리 중... {frame_count}/{total_frames} 프레임 ({progress:.1f}%)")
            
            # 화면 표시 (선택적)
            if show_display:
                # 표시용 이미지 크기 축소 (빠른 표시를 위해)
                display_frame = cv2.resize(result, (width//2, height//2))
                cv2.imshow("Pose-stable Funny Filter (Video) - ESC: 창 닫기", display_frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC 키
                    show_display = False
                    cv2.destroyAllWindows()
                    print("\n창 표시를 중단했습니다. 백그라운드에서 계속 처리 중...\n")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n영상 결과 저장 완료: {out_path}")
