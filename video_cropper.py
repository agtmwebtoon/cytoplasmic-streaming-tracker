import sys
import cv2
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QCheckBox
import numpy as np

sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

class VideoCropGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.video_capture = None
        self.current_frame = None
        self.is_cropping = False
        self.crop_rect = QRect()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Video Crop GUI')
        self.setFixedSize(512, 512)

        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 492, 492)

        self.open_button = QPushButton('Open Video', self)
        self.open_button.setGeometry(10, 512 - 40, 100, 30)
        self.open_button.clicked.connect(self.open_video)

        self.crop_button = QPushButton('Crop Video', self)
        self.crop_button.setGeometry(120, 512 - 40, 100, 30)
        self.crop_button.clicked.connect(self.start_cropping)
        self.crop_button.setEnabled(False)

        self.filter_checkbox = QCheckBox('Apply Canny Filter', self)
        self.filter_checkbox.setGeometry(230, 512 - 40, 150, 30)

    def open_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, 'Open Video File', filter='*.avi')
        if video_path:
            self.video_path = video_path
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.crop_button.setEnabled(True)
            self.display_frame()

    def start_cropping(self):
        self.is_cropping = True
        self.crop_rect = QRect()

    def mousePressEvent(self, event):
        if self.is_cropping and event.button() == Qt.LeftButton:
            self.crop_rect.setTopLeft(event.pos())
            self.crop_rect.setBottomRight(event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_cropping:
            self.crop_rect.setBottomRight(event.pos())
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.is_cropping:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(self.crop_rect)

    def mouseReleaseEvent(self, event):
        if self.is_cropping and event.button() == Qt.LeftButton:
            self.is_cropping = False
            print(self.crop_rect.top())
            print(self.crop_rect.bottom())
            print(self.crop_rect.right())
            print(self.crop_rect.left())
            self.update()
            self.crop_video()

    def crop_video(self):
        save_dialog = QFileDialog()
        save_path, _ = save_dialog.getSaveFileName(self, 'Save Cropped Video', filter='*.avi')
        if save_path:
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (self.crop_rect.width(), self.crop_rect.height()))

            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                cropped_frame = frame[self.crop_rect.top() * 4:self.crop_rect.bottom() * 4,
                                      self.crop_rect.left() * 4:self.crop_rect.right() * 4]
                cropped_frame = cv2.resize(cropped_frame, (self.crop_rect.width(), self.crop_rect.height()))

                if self.filter_checkbox.isChecked():
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    cropped_frame = cv2.filter2D(cropped_frame, -1, sharpening_mask2)
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_GRAY2RGB)

                video_writer.write(cropped_frame)

            video_writer.release()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video capture to the beginning
            self.display_frame()

    def display_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = QImage(self.current_frame,
                         self.current_frame.shape[1],
                         self.current_frame.shape[0],
                         QImage.Format_RGB888)
            pix_map = QPixmap.fromImage(img)
            pix_map = pix_map.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(pix_map)

    def closeEvent(self, event):
        if self.video_capture:
            self.video_capture.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoCropGUI()
    window.show()
    sys.exit(app.exec_())
