# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'AR_Filter_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.9.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QScrollArea,
    QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(752, 329)
        MainWindow.setAnimated(True)
        self.action_load_media = QAction(MainWindow)
        self.action_load_media.setObjectName(u"action_load_media")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.video_display_label = QLabel(self.centralwidget)
        self.video_display_label.setObjectName(u"video_display_label")
        self.video_display_label.setAutoFillBackground(True)
        self.video_display_label.setScaledContents(True)

        self.horizontalLayout_3.addWidget(self.video_display_label)

        self.brt_box = QWidget(self.centralwidget)
        self.brt_box.setObjectName(u"brt_box")
        self.menu_toggle_button = QPushButton(self.brt_box)
        self.menu_toggle_button.setObjectName(u"menu_toggle_button")
        self.menu_toggle_button.setEnabled(True)
        self.menu_toggle_button.setGeometry(QRect(7, 10, 40, 40))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.menu_toggle_button.sizePolicy().hasHeightForWidth())
        self.menu_toggle_button.setSizePolicy(sizePolicy)
        self.menu_toggle_button.setMinimumSize(QSize(30, 30))
        self.menu_toggle_button.setMaximumSize(QSize(16777215, 16777215))
        self.menu_toggle_button.setStyleSheet(u"border-radius: 15px;\n"
"background-color: white;")
        self.menu_toggle_button.setCheckable(True)
        self.filter_scroll_area = QScrollArea(self.brt_box)
        self.filter_scroll_area.setObjectName(u"filter_scroll_area")
        self.filter_scroll_area.setGeometry(QRect(0, 60, 81, 181))
        sizePolicy.setHeightForWidth(self.filter_scroll_area.sizePolicy().hasHeightForWidth())
        self.filter_scroll_area.setSizePolicy(sizePolicy)
        self.filter_scroll_area.setMinimumSize(QSize(30, 0))
        self.filter_scroll_area.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 79, 179))
        self.filter_bnt_06 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_06.setObjectName(u"filter_bnt_06")
        self.filter_bnt_06.setGeometry(QRect(0, 300, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_06.sizePolicy().hasHeightForWidth())
        self.filter_bnt_06.setSizePolicy(sizePolicy)
        self.filter_bnt_06.setMinimumSize(QSize(50, 50))
        self.filter_bnt_06.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_02 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_02.setObjectName(u"filter_bnt_02")
        self.filter_bnt_02.setGeometry(QRect(0, 60, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_02.sizePolicy().hasHeightForWidth())
        self.filter_bnt_02.setSizePolicy(sizePolicy)
        self.filter_bnt_02.setMinimumSize(QSize(50, 50))
        self.filter_bnt_02.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;\n"
"")
        self.filter_bnt_03 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_03.setObjectName(u"filter_bnt_03")
        self.filter_bnt_03.setGeometry(QRect(0, 120, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_03.sizePolicy().hasHeightForWidth())
        self.filter_bnt_03.setSizePolicy(sizePolicy)
        self.filter_bnt_03.setMinimumSize(QSize(50, 50))
        self.filter_bnt_03.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_05 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_05.setObjectName(u"filter_bnt_05")
        self.filter_bnt_05.setGeometry(QRect(0, 240, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_05.sizePolicy().hasHeightForWidth())
        self.filter_bnt_05.setSizePolicy(sizePolicy)
        self.filter_bnt_05.setMinimumSize(QSize(50, 50))
        self.filter_bnt_05.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_04 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_04.setObjectName(u"filter_bnt_04")
        self.filter_bnt_04.setGeometry(QRect(0, 180, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_04.sizePolicy().hasHeightForWidth())
        self.filter_bnt_04.setSizePolicy(sizePolicy)
        self.filter_bnt_04.setMinimumSize(QSize(50, 50))
        self.filter_bnt_04.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_01 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_01.setObjectName(u"filter_bnt_01")
        self.filter_bnt_01.setGeometry(QRect(0, 0, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_01.sizePolicy().hasHeightForWidth())
        self.filter_bnt_01.setSizePolicy(sizePolicy)
        self.filter_bnt_01.setMinimumSize(QSize(50, 50))
        self.filter_bnt_01.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_07 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_07.setObjectName(u"filter_bnt_07")
        self.filter_bnt_07.setGeometry(QRect(0, 360, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_07.sizePolicy().hasHeightForWidth())
        self.filter_bnt_07.setSizePolicy(sizePolicy)
        self.filter_bnt_07.setMinimumSize(QSize(50, 50))
        self.filter_bnt_07.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_08 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_08.setObjectName(u"filter_bnt_08")
        self.filter_bnt_08.setGeometry(QRect(0, 420, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_08.sizePolicy().hasHeightForWidth())
        self.filter_bnt_08.setSizePolicy(sizePolicy)
        self.filter_bnt_08.setMinimumSize(QSize(50, 50))
        self.filter_bnt_08.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_09 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_09.setObjectName(u"filter_bnt_09")
        self.filter_bnt_09.setGeometry(QRect(0, 480, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_09.sizePolicy().hasHeightForWidth())
        self.filter_bnt_09.setSizePolicy(sizePolicy)
        self.filter_bnt_09.setMinimumSize(QSize(50, 50))
        self.filter_bnt_09.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_10 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_10.setObjectName(u"filter_bnt_10")
        self.filter_bnt_10.setGeometry(QRect(0, 540, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_10.sizePolicy().hasHeightForWidth())
        self.filter_bnt_10.setSizePolicy(sizePolicy)
        self.filter_bnt_10.setMinimumSize(QSize(50, 50))
        self.filter_bnt_10.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_11 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_11.setObjectName(u"filter_bnt_11")
        self.filter_bnt_11.setGeometry(QRect(0, 600, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_11.sizePolicy().hasHeightForWidth())
        self.filter_bnt_11.setSizePolicy(sizePolicy)
        self.filter_bnt_11.setMinimumSize(QSize(50, 50))
        self.filter_bnt_11.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_12 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_12.setObjectName(u"filter_bnt_12")
        self.filter_bnt_12.setGeometry(QRect(0, 660, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_12.sizePolicy().hasHeightForWidth())
        self.filter_bnt_12.setSizePolicy(sizePolicy)
        self.filter_bnt_12.setMinimumSize(QSize(50, 50))
        self.filter_bnt_12.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_13 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_13.setObjectName(u"filter_bnt_13")
        self.filter_bnt_13.setGeometry(QRect(0, 720, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_13.sizePolicy().hasHeightForWidth())
        self.filter_bnt_13.setSizePolicy(sizePolicy)
        self.filter_bnt_13.setMinimumSize(QSize(50, 50))
        self.filter_bnt_13.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_14 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_14.setObjectName(u"filter_bnt_14")
        self.filter_bnt_14.setGeometry(QRect(0, 780, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_14.sizePolicy().hasHeightForWidth())
        self.filter_bnt_14.setSizePolicy(sizePolicy)
        self.filter_bnt_14.setMinimumSize(QSize(50, 50))
        self.filter_bnt_14.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_15 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_15.setObjectName(u"filter_bnt_15")
        self.filter_bnt_15.setGeometry(QRect(0, 840, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_15.sizePolicy().hasHeightForWidth())
        self.filter_bnt_15.setSizePolicy(sizePolicy)
        self.filter_bnt_15.setMinimumSize(QSize(50, 50))
        self.filter_bnt_15.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_16 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_16.setObjectName(u"filter_bnt_16")
        self.filter_bnt_16.setGeometry(QRect(0, 900, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_16.sizePolicy().hasHeightForWidth())
        self.filter_bnt_16.setSizePolicy(sizePolicy)
        self.filter_bnt_16.setMinimumSize(QSize(50, 50))
        self.filter_bnt_16.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_17 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_17.setObjectName(u"filter_bnt_17")
        self.filter_bnt_17.setGeometry(QRect(0, 960, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_17.sizePolicy().hasHeightForWidth())
        self.filter_bnt_17.setSizePolicy(sizePolicy)
        self.filter_bnt_17.setMinimumSize(QSize(50, 50))
        self.filter_bnt_17.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_18 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_18.setObjectName(u"filter_bnt_18")
        self.filter_bnt_18.setGeometry(QRect(0, 1020, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_18.sizePolicy().hasHeightForWidth())
        self.filter_bnt_18.setSizePolicy(sizePolicy)
        self.filter_bnt_18.setMinimumSize(QSize(50, 50))
        self.filter_bnt_18.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_bnt_19 = QPushButton(self.scrollAreaWidgetContents)
        self.filter_bnt_19.setObjectName(u"filter_bnt_19")
        self.filter_bnt_19.setGeometry(QRect(0, 1080, 50, 50))
        sizePolicy.setHeightForWidth(self.filter_bnt_19.sizePolicy().hasHeightForWidth())
        self.filter_bnt_19.setSizePolicy(sizePolicy)
        self.filter_bnt_19.setMinimumSize(QSize(50, 50))
        self.filter_bnt_19.setStyleSheet(u"border-radius: 25px; /* \ud06c\uae30\uc758 \uc808\ubc18 */\n"
"background-color: white;")
        self.filter_scroll_area.setWidget(self.scrollAreaWidgetContents)

        self.horizontalLayout_3.addWidget(self.brt_box)

        self.horizontalLayout_3.setStretch(0, 9)
        self.horizontalLayout_3.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 752, 33))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.action_load_media)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"AR Filter", None))
        self.action_load_media.setText(QCoreApplication.translate("MainWindow", u"\uc601\uc0c1/\uc0ac\uc9c4 \ubd88\ub7ec\uc624\uae30", None))
        self.video_display_label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.menu_toggle_button.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_06.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_02.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_03.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_05.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_04.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_01.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_07.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_08.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_09.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_10.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_11.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_12.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_13.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_14.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_15.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_16.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_17.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_18.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.filter_bnt_19.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

