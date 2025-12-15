import cv2
import dlib
import numpy as np
import os
class FastEyeGazeTracker:
    def __init__(self):
        # dlib 얼굴 검출기 및 랜드마크 예측기 초기화
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # 눈 랜드마크 인덱스
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        
        # 스무딩을 위한 변수
        self.prev_left_pupil = None
        self.prev_right_pupil = None
        self.alpha = 0.6
    
    def get_eye_region(self, landmarks, eye_points):
        """눈 영역의 ROI : 원하는 부분만 추출"""


        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        x, y, w, h = cv2.boundingRect(points)


        #마진 넣기 : 동공잘림방지/흰자위 영역을 확보하여 동공 위치 비율 계산에 사용
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.4)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = w + 2 * margin_x
        h = h + 2 * margin_y
        
        return x, y, w, h, points
    
    def detect_pupil_fast(self, eye_roi):
        """빠른 동공 검출"""
        if eye_roi.size == 0 or eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
            return None, None, None
        
        # 흑백 변환
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        
        # 간단한 블러
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu threshold
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 간단한 morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        
        # Contour 찾기
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, None, threshold
        
        # 크기 필터링
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < eye_roi.shape[0] * eye_roi.shape[1] * 0.6:
                valid_contours.append((contour, area))
        
        if len(valid_contours) == 0:
            return None, None, threshold
        
        pupil_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        M = cv2.moments(pupil_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, threshold
        
        return None, None, threshold
    
    def smooth_pupil_position(self, current_pupil, prev_pupil):
        """동공 위치 스무딩"""
        if prev_pupil is None or current_pupil is None:
            return current_pupil
        
        cx_curr, cy_curr = current_pupil
        cx_prev, cy_prev = prev_pupil
        
        cx = int(self.alpha * cx_prev + (1 - self.alpha) * cx_curr)
        cy = int(self.alpha * cy_prev + (1 - self.alpha) * cy_curr)
        
        return (cx, cy)
    
    def estimate_gaze_direction(self, cx, eye_width):
        """시선 방향 추정"""
        if cx is None or eye_width == 0:
            return "Unknown", 0.5
        
        ratio = cx / eye_width
        
        if ratio < 0.4:
            return "Left", ratio
        elif ratio > 0.6:
            return "Right", ratio
        else:
            return "Center", ratio
    
    def process_frame(self, frame):
        """프레임 처리 및 시선 추적"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출 (upsample=0으로 속도 향상)
        faces = self.detector(gray_frame, 0)
        
        if len(faces) == 0:
            return frame, None
        
        face = faces[0]
        landmarks = self.predictor(gray_frame, face)
        
        # 왼쪽 눈
        lx, ly, lw, lh, left_points = self.get_eye_region(landmarks, self.LEFT_EYE)
        left_eye_roi = frame[ly:ly+lh, lx:lx+lw]
        left_cx_raw, left_cy_raw, _ = self.detect_pupil_fast(left_eye_roi)
        
        if left_cx_raw is not None:
            left_pupil = self.smooth_pupil_position((left_cx_raw, left_cy_raw), 
                                                    self.prev_left_pupil)
            self.prev_left_pupil = left_pupil
            left_cx, left_cy = left_pupil
        else:
            left_cx, left_cy = None, None
        
        # 오른쪽 눈
        rx, ry, rw, rh, right_points = self.get_eye_region(landmarks, self.RIGHT_EYE)
        right_eye_roi = frame[ry:ry+rh, rx:rx+rw]
        right_cx_raw, right_cy_raw, _ = self.detect_pupil_fast(right_eye_roi)
        
        if right_cx_raw is not None:
            right_pupil = self.smooth_pupil_position((right_cx_raw, right_cy_raw), 
                                                     self.prev_right_pupil)
            self.prev_right_pupil = right_pupil
            right_cx, right_cy = right_pupil
        else:
            right_cx, right_cy = None, None
        
        # 시선 방향
        left_gaze, left_ratio = self.estimate_gaze_direction(left_cx, lw)
        right_gaze, right_ratio = self.estimate_gaze_direction(right_cx, rw)
        
        if left_gaze == right_gaze:
            final_gaze = left_gaze
        elif left_gaze == "Unknown" and right_gaze != "Unknown":
            final_gaze = right_gaze
        elif right_gaze == "Unknown" and left_gaze != "Unknown":
            final_gaze = left_gaze
        else:
            final_gaze = "Center"
        
        # 시각화
        result_frame = frame.copy()
        
        # 눈 영역 박스
        cv2.rectangle(result_frame, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 1)
        cv2.rectangle(result_frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 1)
        
        # 동공 중심
        if left_cx is not None and left_cy is not None:
            cv2.circle(result_frame, (lx + left_cx, ly + left_cy), 4, (0, 0, 255), -1)
        
        if right_cx is not None and right_cy is not None:
            cv2.circle(result_frame, (rx + right_cx, ry + right_cy), 4, (0, 0, 255), -1)
        
        # 시선 방향
        gaze_color = (0, 255, 0) if final_gaze == "Center" else (0, 255, 255)
        cv2.putText(result_frame, f"Gaze: {final_gaze}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, gaze_color, 2)
        
        return result_frame, final_gaze


def main():
    input_path = "smile_girl3.mp4"
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return
    
    tracker = FastEyeGazeTracker()
    
    print("눈동자 추적 시작... 'esc'를 눌러 종료")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 프레임 처리
        result_frame, gaze_direction = tracker.process_frame(frame)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.splitext(input_path)[0] + "_tracker.mp4" 
        out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        
        # 결과 표시
        cv2.imshow("Eye Gaze Tracking", result_frame)
        out.write(result_frame)
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    out.release()

if __name__ == "__main__":
    main()