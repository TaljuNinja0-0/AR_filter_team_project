import sys
import os
# PySide6 모듈
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QMessageBox)
from PySide6.QtCore import (QTimer, Qt, QDateTime, QSize)
from PySide6.QtGui import (QImage, QPixmap, QIcon)

# OpenCV 및 Dlib 모듈 (필수)
import cv2 
# import dlib  # Dlib은 AR 로직 함수 내에서 사용.
import numpy as np

# 변환된 UI 파일에서 클래스를 import. (ui_AR_Filter.py)
from ui_AR_Filter import Ui_MainWindow

# filter import
from Distort_augmented_filter import apply_filter as Distorted_filter # 왜곡 필터
from Glitch_filter_mapping import apply_filter as Glitch_filter # 글리치 필터
from camera_filter import apply_camera_filter as camera_filter # 카메라 필터
from vintage_filter import apply_vintage_filter as vintage_filter # 빈티지 필터
from glasses_filter_mapping import apply_glasses_filter as glasses_filter # 안경 필터
from monocle_filter_mapping import apply_monocle_filter as monocle_filter # 모노클 필터
from dog_filter_mapping import apply_dog_filter as dog_filter # 강아지 필터
from cat_filter_mapping import apply_cat_filter as cat_filter # 고양이 필터
from bunny_filter_mapping import apply_bunny_filter as bunny_filter # 토끼 필터
from carnivalmask_filter_mapping import apply_carnivalmask_filter as carnivalmask_filter # 무대 가면 필터
from mustache_filter import apply_mustache_filter as mustache_filter # 콧수염 필터
from whitemask_filter import apply_whitemask_filter as whitemask_filter # 흰 마스크 필터
from santa_filter import apply_santa_filter_optimized as santa_filter # 산타 필터
from sunglass_filter_mapping import apply_sunglass_filter as sunglasses_filter # 선글라스 필터
from mask_filter import apply_mask_filter as mask_filter # 마스크 필터  


# 에플리케이션 메인 윈도우 클래스
class ARFilterApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

        # === 상태 변수 초기화 ===
        self.cap = None  # cv2.VideoCapture 객체
        self.current_filter = None # 현재 적용 필터 (예: "filter1", "none")
        self.media_type = None  # 'video' 또는 'image'
        self.loaded_file_path = None  # 불러온 파일 경로
        self.video_writer = None  # 영상 저장용 VideoWriter
        self.is_recording = False  # 녹화 중인지 여부
        self.recorded_frames = []  # 녹화된 프레임 저장
        self.current_frame = None  # 사진용 현재 프레임 저장
        
        # UI 로드 (디자인 적용)
        self.setupUi(self)
        
        self.setWindowTitle("AR Filter")

        # 윈도우 아이콘 설정
        icon_path = "icons/Icon_app.png"

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            print(f"윈도우 아이콘 설정됨: {icon_path}")
        else:
            print(f"경고: 아이콘 파일 '{icon_path}'을 찾을 수 없습니다. 기본 아이콘을 사용합니다.")

        # 윈도우 창 최대화
        self.showMaximized() # 사용자가 크기 조절 가능.


        # QPushButton (필터 버튼)을 필터링.
        self.filter_buttons = {
            self.filter_bnt_01: "filter_bnt_01",
            self.filter_bnt_02: "filter_bnt_02",
            self.filter_bnt_03: "filter_bnt_03",
            self.filter_bnt_04: "filter_bnt_04",
            self.filter_bnt_05: "filter_bnt_05",
            self.filter_bnt_06: "filter_bnt_06",
            self.filter_bnt_07: "filter_bnt_07",
            self.filter_bnt_08: "filter_bnt_08",
            self.filter_bnt_09: "filter_bnt_09",
            self.filter_bnt_10: "filter_bnt_10",
            self.filter_bnt_11: "filter_bnt_11",
            self.filter_bnt_12: "filter_bnt_12",
            self.filter_bnt_13: "filter_bnt_13",
            self.filter_bnt_14: "filter_bnt_14",
            self.filter_bnt_15: "filter_bnt_15",
            self.filter_bnt_16: "filter_bnt_16",
            self.filter_bnt_17: "filter_bnt_17",
            self.filter_bnt_18: "filter_bnt_18",
            self.filter_bnt_19: "filter_bnt_19",
            }
        
         # 필터 이름 매핑
        self.filter_names = {
            "filter_bnt_01": "Distorted_filter",
            "filter_bnt_02": "Glitch_filter",
            "filter_bnt_03": "camera_filter",
            "filter_bnt_04": "vintage_filter",
            "filter_bnt_05": "glasses_filter",
            "filter_bnt_06": "monocle_filter",
            "filter_bnt_07": "dog_filter",
            "filter_bnt_08": "cat_filter",
            "filter_bnt_09": "bunny_filter",
            "filter_bnt_10": "carnivalmask_filter",
            "filter_bnt_11": "mustache_filter",
            "filter_bnt_12": "whitemask_filter",
            "filter_bnt_13": "santa_filter",
            "filter_bnt_14": "sunglasses_filter",
            "filter_bnt_15": "mask_filter",
            "filter_bnt_16": "불",
            "filter_bnt_17": "시선 트랙킹",
            "filter_bnt_18": "잔상",
            "filter_bnt_19": "하트"
        }
        
        self.filter_icons = { # 아이콘 매핑
            "filter_bnt_01": "icons/filter_01.png",  # 왜곡 필터 아이콘
            "filter_bnt_02": "icons/filter_02.png",  # 글리치
            "filter_bnt_03": "icons/filter_03.png",  # 카메라
            "filter_bnt_04": "icons/filter_04.png",  # 빈티지
            "filter_bnt_05": "icons/filter_05.png",  # 안경
            "filter_bnt_06": "icons/filter_06.png",  # 모노클
            "filter_bnt_07": "icons/filter_07.png",  # 강아지
            "filter_bnt_08": "icons/filter_08.png",  # 고양이
            "filter_bnt_09": "icons/filter_09.png",  # 토끼 
            "filter_bnt_10": "icons/filter_10.png",  # 무대 가면
            "filter_bnt_11": "icons/filter_11.png",  # 콧수염
            "filter_bnt_12": "icons/filter_12.png",  # 흰 마스크
            "filter_bnt_13": "icons/filter_13.png",  # 산타
            "filter_bnt_14": "icons/filter_14.png",  # 선글라스
            "filter_bnt_15": "icons/filter_15.png",  # 마스크
            "filter_bnt_16": "icons/filter_16.png",  # 불
            "filter_bnt_17": "icons/filter_17.png",  # 시선 트랙킹
            "filter_bnt_18": "icons/filter_18.png",  # 잔상 필터
            "filter_bnt_19": "icons/filter_19.png",  # 하트 필터
        }

        # === UI  ===
        self.fix_ui_issues()
        
        # === 초기 UI 설정 ===
        self.filter_scroll_area.setVisible(False) 
        self.video_display_label.setText("")
        
        # === 이벤트 연결 ===
        
        # 메뉴바: 영상/사진 불러오기 액션 연결
        self.action_load_media.triggered.connect(self.load_media_file)

        # 메뉴 토글 버튼 연결
        self.menu_toggle_button.toggled.connect(self.toggle_filter_menu)

        # 필터 버튼 연결 
        for button in self.filter_buttons:
            filter_id = button.objectName()
            button.clicked.connect(lambda checked, fid=filter_id: self.select_filter(fid))
        
        # === OpenCV 타이머 설정 ===
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # self.timer.start(30) 은 미디어를 로드할 때 시작합니다.

        # === 필터 설정 ===
        # 각 필터별 함수 연결
        self.filter_map = {
            "filter_bnt_01": Distorted_filter, # 왜곡
            "filter_bnt_02": Glitch_filter, # 그리치
            "filter_bnt_03": camera_filter, # 카메라
            "filter_bnt_04": vintage_filter, # 빈티지
            "filter_bnt_05": glasses_filter, # 안경
            "filter_bnt_06": monocle_filter, # 모노클
            "filter_bnt_07": dog_filter, # 강아지
            "filter_bnt_08": cat_filter, # 고양이
            "filter_bnt_09": bunny_filter, # 토끼
            "filter_bnt_10": carnivalmask_filter, # 무대 가면
            "filter_bnt_11": mustache_filter, # 콧수염
            "filter_bnt_12": whitemask_filter, # 흰 마스크
            "filter_bnt_13": santa_filter, # 산타 필터
            "filter_bnt_14": sunglasses_filter, # 선글라스
            "filter_bnt_15": mask_filter, # 마스크
            # "filter_bnt_02": other_filter,
        }
        
        # 각 필터별 사용 가능한 미디어 타입 설정
        self.filter_media_support = {
            "filter_bnt_01": ['image', 'video'],  # Distorted_filter
            "filter_bnt_02": ['image', 'video'],  # Glitch_filter
            "filter_bnt_03": ['video'],  # camera_filter
            "filter_bnt_04": ['video'],  # vintage_filter
            "filter_bnt_05": ['image', 'video'],  # 안경
            "filter_bnt_06": ['image', 'video'],  # 모노클
            "filter_bnt_07": ['image', 'video'],  # 강아지
            "filter_bnt_08": ['image', 'video'],  # 고양이
            "filter_bnt_09": ['image', 'video'],  # 토끼 
            "filter_bnt_10": ['image', 'video'],  # 무대 가면
            "filter_bnt_11": ['image', 'video'],  # 콧수염
            "filter_bnt_12": ['image', 'video'],  # 흰 마스크
            "filter_bnt_13": ['image', 'video'],  # 산타 필터
            "filter_bnt_14": ['image', 'video'],  # 선글라스
            "filter_bnt_15": ['image', 'video'],  # 마스크
            "filter_bnt_16": ['image', 'video'], 
            "filter_bnt_17": ['image'],
            "filter_bnt_18": ['video'],
            "filter_bnt_19": ['video']
        }


    def fix_ui_issues(self):
        # 비디오 비율 유지
        self.video_display_label.setScaledContents(False)
        # 중앙 정렬
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 라벨 최소 크기 제한 X
        self.video_display_label.setMinimumSize(QSize(1, 1))
        
        # 스크롤 영역의 최소 너비 설정
        self.filter_scroll_area.setMinimumWidth(70)
        self.filter_scroll_area.setMaximumWidth(90)
        
        # 스크롤 내용 위젯의 레이아웃을 VBoxLayout으로 재구성
        content_widget = self.filter_scroll_area.widget()
        
        # 기존 버튼들을 가져오기
        buttons = [
            child for child in content_widget.children() 
            if isinstance(child, QPushButton) and child.objectName().startswith('filter_')
        ]
        
        # 버튼들을 이름 순서대로 정렬
        buttons.sort(key=lambda btn: btn.objectName())
        
        # 새로운 레이아웃 생성
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(10)  # 버튼 간격
        layout.setContentsMargins(10, 10, 10, 10)
        
        for button in buttons:
            layout.addWidget(button)
        
        # 스트레치 추가 
        layout.addStretch()
        
        # 스크롤 영역 설정
        self.filter_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.filter_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # brt_box 최소 크기 설정
        self.brt_box.setMinimumWidth(90)
        self.brt_box.setMaximumWidth(110)
        
        # === 버튼 이미지 설정 ===
        self.setup_filter_button_images()
    
    def setup_filter_button_images(self):
        # 필터 버튼에 아이콘 이미지 설정
        for button in self.filter_buttons:
            button_name = button.objectName()
            
            if button_name in self.filter_icons:
                icon_path = self.filter_icons[button_name]
                
                # 이미지 파일이 존재하는지 확인
                if os.path.exists(icon_path):
                    icon = QIcon(icon_path)
                    button.setIcon(icon)
                    button.setIconSize(QSize(40, 40))  # 아이콘 크기
                    button.setText("")  # 텍스트 제거
                else:
                    # 이미지가 없으면 버튼 번호만 표시
                    button_number = button_name.split('_')[-1]
                    button.setText(button_number)



    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        # 중앙 위젯의 높이 가져오기
        available_height = self.centralwidget.height()
        top_margin = 60  # 버튼 아래 시작 위치
        bottom_margin = 20  # 하단 여백
       # button_height = self.menu_toggle_button.height()
        
        # 스크롤 영역 높이 계산
        new_height = available_height - top_margin - bottom_margin
        
        # 최소 높이 보장
        if new_height < 100:
            new_height = 100
        
        # 스크롤 영역 크기 조정
        current_geometry = self.filter_scroll_area.geometry()
        self.filter_scroll_area.setGeometry(
            current_geometry.x(),
            top_margin,
            current_geometry.width(),
            new_height
        )

        if self.media_type == 'image' and self.current_frame is not None:
            self.update_image_display()

    def update_filter_buttons_state(self):
        """현재 미디어 타입에 따라 필터 버튼 활성화/비활성화"""
        if self.media_type is None:
            # 미디어가 로드되지 않은 경우 모든 버튼 비활성화
            for button in self.filter_buttons:
                button.setEnabled(False)
                button.setStyleSheet("border-radius: 25px; background-color: #CCCCCC;")  # 회색
            return
        
        for button in self.filter_buttons:
            button_name = button.objectName()
            
            # 이 버튼이 현재 미디어 타입을 지원하는지 확인
            if button_name in self.filter_media_support:
                supported_types = self.filter_media_support[button_name]
                
                if self.media_type in supported_types:
                    # 활성화
                    button.setEnabled(True)
                    # 현재 선택된 필터면 강조
                    if button_name == self.current_filter:
                        button.setStyleSheet("""
                            border-radius: 25px;
                            background-color: #87CEEB;
                            border: 3px solid #4682B4;
                        """)
                    else:
                        button.setStyleSheet("border-radius: 25px; background-color: white;")
                else:
                    # 비활성화
                    button.setEnabled(False)
                    button.setStyleSheet("border-radius: 25px; background-color: #CCCCCC;")  # 회색
            else:
                # 설정되지 않은 버튼은 기본적으로 비활성화
                button.setEnabled(False)
                button.setStyleSheet("border-radius: 25px; background-color: #CCCCCC;")

    # --- [로직 함수] ---

    def load_media_file(self):
        # 파일 탐색기에서 영상/사진 불러오기
        if self.cap and self.cap.isOpened():
             self.cap.release()
             self.timer.stop()
             
        file_path, _ = QFileDialog.getOpenFileName(
            self, "영상/사진 파일 선택", "", 
            "미디어 파일 (*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp);;카메라 (*)"
        )
        
        if file_path:
            self.loaded_file_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            
            # 영상 vs 사진 판별
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov']:
                self.media_type = 'video'
                print(f"영상 로드됨: {file_path}")
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.media_type = 'image'
                print(f"사진 로드됨: {file_path}")
            else:
                self.media_type = 'unknown'
        else:
            self.cap = cv2.VideoCapture(0) # 카메라 시도
            self.media_type = 'video'
            self.loaded_file_path = None
            print("카메라 모드")

        if self.cap.isOpened():
            # 새 미디어 로드 시 필터 자동 해제
            self.current_filter = None

            self.timer.start(30) # 30ms 마다 프레임 업데이트 시작
            print("미디어 로드 및 타이머 시작")
            
            # 필터 버튼 상태 업데이트
            self.update_filter_buttons_state()
        else:
            print("미디어 파일을 열거나 카메라에 접근할 수 없습니다.")
            self.video_display_label.setText("미디어를 로드할 수 없음")
            
    
    def toggle_filter_menu(self, is_checked):
        # 버튼 클릭 > 필터 버튼 띄움 > 버튼 한번 더 클릭 > 캡쳐 및 닫힘
        if is_checked:
            # 메뉴 열림 (버튼이 눌린 상태): 스크롤 영역 보이기
            self.filter_scroll_area.setVisible(True)
            print("필터 메뉴 열림")
            
        else:
            # 메뉴 닫힘 (버튼이 풀린 상태): 스크롤 영역 숨기기 + 캡처/녹화
            self.filter_scroll_area.setVisible(False)
            print("필터 메뉴 닫힘 및 캡처/녹화 시도")

            # 필터가 선택되지 않았으면 저장/캡쳐 안 함
            if self.current_filter is None:
                QMessageBox.warning(
                    self,
                    "필터 미선택",
                    "필터를 선택한 후 저장/캡쳐할 수 있습니다."
                )
                return
            
            # 미디어 타입에 따라 캡처 또는 저장
            if self.media_type == 'image':
                self.capture_photo()
            elif self.media_type == 'video':
                self.save_entire_video()
        
    def select_filter(self, filter_name):
        #필터 버튼 클릭 시 현재 필터 적용 / 메뉴 열린 상태 유지
        # 같은 필터를 다시 누르면 해제
        if self.current_filter == filter_name:
            self.current_filter = None
            print(f"필터 해제됨: {filter_name}")
        else:
            self.current_filter = filter_name
            print(f"필터 선택됨: {filter_name}")
        
        # 버튼 상태 업데이트 (선택된 버튼 강조)
        self.update_filter_buttons_state()
        
        # 사진 모드일 때 즉시 화면 업데이트
        if self.media_type == 'image' and self.current_frame is not None:
            self.update_image_display()

    def update_image_display(self):
        # 사진 모드에서 필터 적용 후 즉시 화면 업데이트
        if self.current_frame is None:
            return
        
        # 필터 적용
        processed_frame = self.apply_ar_filter(self.current_frame.copy(), self.current_filter)
        
        # 화면에 표시
        height, width, channel = processed_frame.shape
        bytes_per_line = 3 * width
        
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = q_pixmap.scaled(
            self.video_display_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_display_label.setPixmap(scaled_pixmap)

    def capture_photo(self):
        # 현재 프레임을 캡처하여 파일로 저장 (사진 모드)
        if self.current_frame is None: 
            print("캡처할 프레임이 없습니다.")
            QMessageBox.warning(self, "캡처 실패")
            return
        
        processed_frame = self.apply_ar_filter(self.current_frame.copy(), self.current_filter)
        
        original_filename = os.path.basename(self.loaded_file_path)
        filename_without_ext = os.path.splitext(original_filename)[0]
        original_ext = os.path.splitext(original_filename)[1]
        
        filter_name = self.filter_names.get(self.current_filter, "필터")
        
        # 새 파일명 생성
        filename = f"{filename_without_ext}_{filter_name}{original_ext}"
        
        # 파일 저장
        success = cv2.imwrite(filename, processed_frame)
        
        if success:
            print(f"사진 저장됨: {filename}")
            
            # 저장 완료 팝업
            QMessageBox.information(
                self,
                "저장 완료",
                f"사진이 저장되었습니다.\n파일명: {filename}"
            )
        else:
            print(f"사진 저장 실패: {filename}")
            QMessageBox.warning(
                self,
                "저장 실패",
                f"사진 저장에 실패했습니다."
            )

    def save_entire_video(self):
        """불러온 영상 전체에 필터를 적용하여 저장 (영상 모드)"""
        if self.cap is None or not self.cap.isOpened():
            print("저장할 영상이 없습니다.")
            QMessageBox.warning(self, "저장 실패", "저장할 영상이 없습니다.")
            return
        
        if self.loaded_file_path is None:
            print("카메라 모드에서는 전체 영상 저장이 불가능합니다.")
            QMessageBox.warning(
                self,
                "저장 불가",
                "카메라 모드에서는 전체 영상 저장이 불가능합니다."
            )
            return
        
        original_filename = os.path.basename(self.loaded_file_path)
        filename_without_ext = os.path.splitext(original_filename)[0]
        
        filter_name = self.filter_names.get(self.current_filter, "필터")
        # 새 파일 생성
        filename = f"{filename_without_ext}_{filter_name}.mp4"
            
        # 저장 중 팝업 표시 (non-blocking)
        progress_msg = QMessageBox(self)
        progress_msg.setWindowTitle("저장 중")
        progress_msg.setText("영상 전체에 필터를 적용하는 중입니다.\n잠시만 기다려주세요...")
        progress_msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
        progress_msg.setModal(False)
        progress_msg.show()
        
        # UI 업데이트 강제 실행
        QApplication.processEvents()
        
        print("영상 전체에 필터 적용 중... 잠시만 기다려주세요.")
        
        # 현재 재생 위치 저장
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
        # 영상 정보 가져오기
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # VideoWriter 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # 처음부터 다시 읽기
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 변수 초기화
        frame_count = 0
        last_progress = -1
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 필터 적용
            processed_frame = self.apply_ar_filter(frame.copy(), self.current_filter)
            
            # 저장
            out.write(processed_frame)
            
            frame_count += 1
            
            # 진행 상황 업데이트
            current_progress = int((frame_count / total_frames) * 100)
            if current_progress != last_progress and current_progress % 10 == 0:
                progress_msg.setText(
                    f"필터를 적용하는 중입니다."
                )
                QApplication.processEvents()
                last_progress = current_progress
                print(f"진행 중... {current_progress}%")
        
        out.release()
        
        # 원래 위치로 복원
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        # 저장 중 팝업 닫기 (강제로 숨기고 삭제)
        progress_msg.hide()
        progress_msg.deleteLater()
        QApplication.processEvents()  # UI 업데이트 강제 실행
        
        print(f"영상 저장 완료: {filename}")
        print(f"총 {frame_count} 프레임 처리됨")
        
        # 저장 완료 팝업
        QMessageBox.information(
            self,
            "저장 완료",
            f"영상이 저장되었습니다.\n파일명: {filename}"
        )

    def start_recording(self):
        # 영상 녹화 시작(영상 모드)
        self.is_recording = True
        self.recorded_frames = []
        print("녹화 시작")
        
    def stop_recording(self):
        """영상 녹화를 중지하고 파일로 저장합니다. (영상 모드)"""
        if not self.is_recording or len(self.recorded_frames) == 0:
            print("녹화된 프레임이 없습니다.")
            return
        
        self.is_recording = False
        
        # 파일명 생성
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        filename = f"recorded_{timestamp}.mp4"
        
        # 첫 프레임의 크기를 가져옴
        height, width = self.recorded_frames[0].shape[:2]
        
        # VideoWriter 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # 프레임 레이트
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # 모든 프레임 저장
        for frame in self.recorded_frames:
            out.write(frame)
        
        out.release()
        self.recorded_frames = []
        
        print(f"영상 저장됨: {filename}")

    def apply_ar_filter(self, frame, filter_name):
        """AR 필터를 적용"""
        if filter_name in self.filter_map:
            return self.filter_map[filter_name](frame)
        
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        
        if ret:
            # 사진 모드일 때 현재 프레임 저장 및 초기화
            if self.media_type == 'image':
                self.current_frame = frame.copy()
                self.timer.stop()

                self.update_image_display() 
                return

            # 1. AR 필터 적용
            processed_frame = self.apply_ar_filter(frame.copy(), self.current_filter)
            
            # 2. 영상 녹화 중이라면 프레임 저장
            if self.is_recording:
                self.recorded_frames.append(processed_frame.copy())

            # 3. OpenCV (BGR) -> PySide6 (QPixmap) 변환
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            q_pixmap = QPixmap.fromImage(q_image)

            # 4. QLabel에 이미지 표시
            scaled_pixmap = q_pixmap.scaled(
                self.video_display_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_display_label.setPixmap(scaled_pixmap)

        else:
            if self.media_type == 'video' and self.loaded_file_path is not None:
                # 영상 파일이 끝에 도달했을 때 (반복 재생 로직)
                
                if self.is_recording:
                    self.stop_recording()
                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            elif self.media_type == 'video' and self.loaded_file_path is None:
                # 카메라 모드 오류/종료 시
                self.timer.stop()
                print("카메라 스트림 종료 또는 오류 발생")



def run_app():
    app = QApplication(sys.argv)
    window = ARFilterApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()