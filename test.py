import cv2
import dlib
import os

# 1. 모델 파일 경로 확인
model_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(model_path):
    print(f"오류: Dlib 모델 파일 '{model_path}'을 찾을 수 없습니다. 프로젝트 폴더에 다운로드했는지 확인하세요.")
    exit()

# 2. Dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
print("Dlib 초기화 성공.")

 # 3. OpenCV 이미지 로드 (샘플 이미지 파일명으로 변경하세요)
img_path = 'Face.jpg' # 본인의 테스트 이미지 파일명으로 변경
if not os.path.exists(img_path):
    print(f"경고: 테스트 이미지 '{img_path}'를 찾을 수 없습니다. OpenCV 기능만 확인합니다.")
    # 이미지 없으면 임시로 빈 이미지 생성
    img = 255 * (cv2.imread(cv2.samples.findFile("starry_night.jpg")) if cv2.samples.findFile("starry_night.jpg") else cv2.IMREAD_COLOR)
    if img is None:
        print("OpenCV 설치는 되었으나, 샘플 이미지 로드에 실패했습니다. 다음 단계로 넘어가세요.")
        exit()
else:
    img = cv2.imread(img_path)

# 4. OpenCV 이미지 처리 테스트
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("OpenCV 이미지 로드 및 색상 변환 성공.")

# 5. Dlib 얼굴 인식 테스트
faces = detector(gray)
print(f"프레임에서 얼굴 {len(faces)}개 발견.")

# 6. 결과 출력
if len(faces) > 0:
    for face in faces:
        landmarks = predictor(gray, face)
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1) # 랜드마크를 녹색 점으로 표시
            cv2.putText(img, str(i), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1) #랜드마크 번호 표시



cv2.imshow("Setup Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# 2. 입력 동영상 파일 경로 (반드시 실제 동영상 파일명으로 변경하세요)
""" video_path = 'smile_girl.mp4' 
if not os.path.exists(video_path):
    print(f"오류: 동영상 파일 '{video_path}'을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()
# --------------------------

# 3. Dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
print("Dlib 초기화 성공.")

# 4. 동영상 캡처 객체 생성
# 웹캠을 사용하려면 video_path 대신 0 또는 1을 넣습니다.
# 예: cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("오류: 동영상 파일을 열 수 없습니다.")
    exit()

print(f"동영상 파일 '{video_path}' 처리 시작...")

# 5. 프레임 처리 루프 시작
while cap.isOpened():
    # 프레임 읽기: ret는 성공 여부, frame은 이미지 데이터
    ret, frame = cap.read()
    
    # 동영상 끝에 도달했거나 프레임을 읽지 못하면 루프 종료
    if not ret:
        print("동영상 끝에 도달했습니다.")
        break
    
    # --- 핵심 처리 로직 시작 ---
    
    # Dlib은 보통 그레이스케일 이미지로 처리하는 것이 효율적입니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Dlib 얼굴 인식
    # 1은 업샘플링 횟수. 0으로 낮추면 더 빠르게 동작합니다.
    faces = detector(gray, 0)
    
    # 랜드마크 추출 및 그리기
    for face in faces:
        landmarks = predictor(gray, face)
        
        # 68개 랜드마크를 순회하며 화면에 그립니다.
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            # 랜드마크를 녹색 점으로 표시
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
            
    # --- 핵심 처리 로직 종료 ---
    
    # 6. 결과 프레임 표시
    cv2.imshow("AR Filter Test", frame)
    
    # 7. 키 입력 대기 (종료 조건)
    # 'q' 키를 누르거나 창이 닫히면 루프 종료
    # waitKey(1)은 1ms마다 키 입력을 확인하며 실시간 스트리밍 효과를 냅니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. 자원 해제
cap.release()
cv2.destroyAllWindows()
print("동영상 처리 및 테스트 완료.")
"""