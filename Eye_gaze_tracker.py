import cv2
import dlib
import numpy as np
import os

_tracker = None

class EyeLaserTracker:
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

        # 마진 넣기 : 동공잘림방지/흰자위 영역을 확보하여 동공 위치 비율 계산에 사용
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.3)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = w + 2 * margin_x
        h = h + 2 * margin_y
        
        return x, y, w, h, points
    
    def detect_pupil_fast(self, eye_roi):
        """빠른 동공 검출"""
        if eye_roi.size == 0 or eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
            return None, None
        
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
            return None, None
        
        # 크기 필터링
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < eye_roi.shape[0] * eye_roi.shape[1] * 0.6:
                valid_contours.append((contour, area))
        
        if len(valid_contours) == 0:
            return None, None
        
        pupil_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        M = cv2.moments(pupil_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        
        return None, None
    
    def smooth_pupil_position(self, current_pupil, prev_pupil):
        """동공 위치 스무딩"""
        if prev_pupil is None or current_pupil is None:
            return current_pupil
        
        cx_curr, cy_curr = current_pupil
        cx_prev, cy_prev = prev_pupil
        
        cx = int(self.alpha * cx_prev + (1 - self.alpha) * cx_curr)
        cy = int(self.alpha * cy_prev + (1 - self.alpha) * cy_curr)
        
        return (cx, cy)
    
    def calculate_laser_direction(self, pupil_pos, eye_roi_x, eye_roi_y, eye_width, eye_height, frame_width, frame_height):
        """흰자/눈동자 비율 기반 3D 시선 방향 레이저 계산"""
        if pupil_pos is None:
            return None
        
        cx, cy = pupil_pos
        
        # 동공의 실제 화면 좌표
        pupil_x = eye_roi_x + cx
        pupil_y = eye_roi_y + cy
        
        # 눈 영역 내에서 동공의 상대적 위치 (0~1 범위)
        ratio_x = cx / eye_width if eye_width > 0 else 0.5
        ratio_y = cy / eye_height if eye_height > 0 else 0.5
        
        # ⭐ Y축 중심점 보정: 동공이 눈의 약간 위쪽에 위치하는 것이 자연스러움
        center_y = 0.25
        
        # 중심에서의 편차 계산
        # ratio가 0.5(X축), 0.2(Y축)일 때 중심(정면)
        deviation_x = ratio_x - 0.5
        deviation_y = ratio_y - center_y
        
        # ⭐ Y축 가중치 조정: Y축 움직임을 더 강조
        weighted_deviation_x = deviation_x * 1.0  # X축 기본
        weighted_deviation_y = deviation_y * 1.4  # Y축 1.4배 가중치
        
        # 레이저 길이 계산 (정면을 볼수록 짧아짐)
        gaze_distance = np.sqrt(weighted_deviation_x**2 + weighted_deviation_y**2)
        
        # 정면(중심)일 때 짧고, 옆/위/아래를 볼 때 길어짐
        min_length = 1   # 최소 레이저 길이 (정면)
        max_length = 900  # 최대 레이저 길이 (극단적인 방향)
        laser_length = min_length + (gaze_distance * 2.5) * (max_length - min_length)
        
        # 시선 방향 벡터 계산
        direction_x = weighted_deviation_x * 2.0  # 배율 증가로 방향 강조
        direction_y = weighted_deviation_y * 2.0
        
        # 방향 벡터 정규화 후 길이 적용
        magnitude = np.sqrt(direction_x**2 + direction_y**2)
        if magnitude > 0.01:  # 0으로 나누는 것 방지
            norm_x = direction_x / magnitude
            norm_y = direction_y / magnitude
        else:
            # 거의 정면을 볼 때는 약간 전방(아래)으로
            norm_x = 0
            norm_y = 0.3
            laser_length = min_length
        
        # 레이저 끝점 계산
        end_x = int(pupil_x + norm_x * laser_length)
        end_y = int(pupil_y + norm_y * laser_length)
        
        # 화면 경계를 넘어가도록 허용 (레이저가 화면 밖으로 뻗어나감)
        if end_x < 0:
            end_x = -100
        elif end_x > frame_width:
            end_x = frame_width + 100
            
        if end_y < 0:
            end_y = -100
        elif end_y > frame_height:
            end_y = frame_height + 100
        
        # 디버깅 정보 반환
        gaze_info = {
            'ratio_x': ratio_x,
            'ratio_y': ratio_y,
            'deviation_x': deviation_x,
            'deviation_y': deviation_y,
            'weighted_dev_y': weighted_deviation_y,
            'laser_length': laser_length
        }
        
        return (pupil_x, pupil_y), (end_x, end_y), gaze_info
    
    def draw_laser(self, frame, start_pos, end_pos, color=(0, 0, 255), thickness=3):
        """레이저 빔 그리기 (개선된 시각 효과)"""
        if start_pos is None or end_pos is None:
            return
        
        # 메인 레이저 빔
        cv2.line(frame, start_pos, end_pos, color, thickness)
        
        # 빛나는 효과 (더 두꺼운 반투명 레이어)
        overlay = frame.copy()
        cv2.line(overlay, start_pos, end_pos, color, thickness + 6)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # 동공 중심에 발광 효과
        cv2.circle(frame, start_pos, 7, color, -1)
        cv2.circle(frame, start_pos, 10, color, 2)
        
        # 레이저 끝부분에 작은 빛 효과
        cv2.circle(frame, end_pos, 4, color, -1)
    
    def draw_debug_info(self, frame, gaze_info, position, color):
        """디버그 정보 표시 (선택적)"""
        if gaze_info is None:
            return
        
        info_text = [
            f"Ratio X: {gaze_info['ratio_x']:.2f}",
            f"Ratio Y: {gaze_info['ratio_y']:.2f}",
            f"Dev Y: {gaze_info['deviation_y']:.2f} -> {gaze_info['weighted_dev_y']:.2f}",
            f"Length: {int(gaze_info['laser_length'])}"
        ]
        
        y_offset = position
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def process_frame(self, frame, show_debug=False):
        """프레임 처리 및 레이저 효과 (한쪽 눈만 감지되어도 양쪽에서 발사)"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = self.detector(gray_frame, 0)
        
        if len(faces) == 0:
            return frame
        
        face = faces[0]
        landmarks = self.predictor(gray_frame, face)
        
        result_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        left_laser_data = None
        right_laser_data = None
        
        # 왼쪽 눈 영역 추출 (항상)
        lx, ly, lw, lh, _ = self.get_eye_region(landmarks, self.LEFT_EYE)
        left_eye_roi = frame[ly:ly+lh, lx:lx+lw]
        left_cx_raw, left_cy_raw = self.detect_pupil_fast(left_eye_roi)
        
        if left_cx_raw is not None:
            left_pupil = self.smooth_pupil_position((left_cx_raw, left_cy_raw), 
                                                    self.prev_left_pupil)
            self.prev_left_pupil = left_pupil
            
            # 왼쪽 눈 레이저 데이터 계산
            left_laser_data = self.calculate_laser_direction(left_pupil, lx, ly, lw, lh, 
                                                             frame_width, frame_height)
            if show_debug and left_laser_data:
                self.draw_debug_info(result_frame, left_laser_data[2], 30, (0, 255, 255))
        
        # 오른쪽 눈 영역 추출 (항상)
        rx, ry, rw, rh, _ = self.get_eye_region(landmarks, self.RIGHT_EYE)
        right_eye_roi = frame[ry:ry+rh, rx:rx+rw]
        right_cx_raw, right_cy_raw = self.detect_pupil_fast(right_eye_roi)
        
        if right_cx_raw is not None:
            right_pupil = self.smooth_pupil_position((right_cx_raw, right_cy_raw), 
                                                     self.prev_right_pupil)
            self.prev_right_pupil = right_pupil
            
            # 오른쪽 눈 레이저 데이터 계산
            right_laser_data = self.calculate_laser_direction(right_pupil, rx, ry, rw, rh, 
                                                              frame_width, frame_height)
            if show_debug and right_laser_data:
                self.draw_debug_info(result_frame, right_laser_data[2], 130, (255, 255, 0))
        
        # 레이저 그리기 로직
        if left_laser_data is not None and right_laser_data is not None:
            # 양쪽 눈 모두 감지된 경우: 평균 방향 계산
            left_start, left_end, left_info = left_laser_data
            right_start, right_end, right_info = right_laser_data
            
            # 각 눈에서의 방향 벡터 계산
            left_dir_x = left_end[0] - left_start[0]
            left_dir_y = left_end[1] - left_start[1]
            
            right_dir_x = right_end[0] - right_start[0]
            right_dir_y = right_end[1] - right_start[1]
            
            # 평균 방향 벡터 계산
            avg_dir_x = (left_dir_x + right_dir_x) // 2
            avg_dir_y = (left_dir_y + right_dir_y) // 2
            
            # 각 눈에서 평균 방향으로 평행한 끝점 계산
            left_parallel_end = (left_start[0] + avg_dir_x, left_start[1] + avg_dir_y)
            right_parallel_end = (right_start[0] + avg_dir_x, right_start[1] + avg_dir_y)
            
            # 양쪽 눈에서 평행한 레이저
            self.draw_laser(result_frame, left_start, left_parallel_end, color=(255, 0, 0), thickness=3)
            self.draw_laser(result_frame, right_start, right_parallel_end, color=(255, 0, 0), thickness=3)
        
        elif left_laser_data is not None:
            # 왼쪽 눈만 감지된 경우: 왼쪽 눈의 방향을 양쪽에 적용
            left_start, left_end, _ = left_laser_data
            
            # 왼쪽 눈의 방향 벡터
            dir_x = left_end[0] - left_start[0]
            dir_y = left_end[1] - left_start[1]
            
            # 오른쪽 눈 위치 추정 (눈 영역의 중심)
            rx_center = rx + rw // 2
            ry_center = ry + rh // 2
            right_start_estimated = (rx_center, ry_center)
            
            # 양쪽 눈에서 같은 방향으로 레이저
            left_parallel_end = (left_start[0] + dir_x, left_start[1] + dir_y)
            right_parallel_end = (right_start_estimated[0] + dir_x, right_start_estimated[1] + dir_y)
            
            self.draw_laser(result_frame, left_start, left_parallel_end, color=(255, 0, 0), thickness=3)
            self.draw_laser(result_frame, right_start_estimated, right_parallel_end, color=(255, 0, 0), thickness=3)
        
        elif right_laser_data is not None:
            # 오른쪽 눈만 감지된 경우: 오른쪽 눈의 방향을 양쪽에 적용
            right_start, right_end, _ = right_laser_data
            
            # 오른쪽 눈의 방향 벡터
            dir_x = right_end[0] - right_start[0]
            dir_y = right_end[1] - right_start[1]
            
            # 왼쪽 눈 위치 추정 (눈 영역의 중심)
            lx_center = lx + lw // 2
            ly_center = ly + lh // 2
            left_start_estimated = (lx_center, ly_center)
            
            # 양쪽 눈에서 같은 방향으로 레이저
            left_parallel_end = (left_start_estimated[0] + dir_x, left_start_estimated[1] + dir_y)
            right_parallel_end = (right_start[0] + dir_x, right_start[1] + dir_y)
            
            self.draw_laser(result_frame, left_start_estimated, left_parallel_end, color=(255, 0, 0), thickness=3)
            self.draw_laser(result_frame, right_start, right_parallel_end, color=(255, 0, 0), thickness=3)
        
        return result_frame

def apply_eye_laser_filter(frame):
    global _tracker

    if _tracker is None:
        _tracker = EyeLaserTracker()

    return _tracker.process_frame(frame, show_debug=False)

'''
def main():
    input_path = "smile_girl3.mp4"
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return
    
    tracker = EyeLaserTracker()
    
    # 비디오 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.splitext(input_path)[0] + "_laser.mp4" 
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print("3D 시선 추적 레이저 효과 처리 중... ESC를 눌러 종료")
    print("'d' 키: 디버그 정보 토글")
    
    frame_count = 0
    show_debug = False
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 프레임 처리 (디버그 옵션)
        result_frame = tracker.process_frame(frame, show_debug=show_debug)
        
        # 결과 저장 및 표시
        out.write(result_frame)
        cv2.imshow("Eye Laser Effect - 3D Gaze Tracking", result_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"처리된 프레임: {frame_count}")
        
        # 키 입력 처리
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('d'):  # 'd' 키로 디버그 토글
            show_debug = not show_debug
            print(f"디버그 모드: {'ON' if show_debug else 'OFF'}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n완료! 출력 파일: {out_path}")


if __name__ == "__main__":
    main()

'''