import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QRect

class ImageSelector(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()
        self.selected_region = None
        self.start_point = None
        self.end_point = None

        self.initUI()

    def initUI(self):
        # Setup main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Display image
        self.label = QLabel(self)
        self.update_image(self.image)
        layout.addWidget(self.label)

        # Add a button to reset selection
        reset_button = QPushButton('Reset Selection')
        reset_button.clicked.connect(self.reset_selection)
        layout.addWidget(reset_button)

        self.label.mousePressEvent = self.mouse_press_event
        self.label.mouseMoveEvent = self.mouse_move_event
        self.label.mouseReleaseEvent = self.mouse_release_event

        self.setWindowTitle('Image Region Selector')
        self.setGeometry(100, 100, 800, 600)
        self.show()

    def update_image(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()

    def mouse_move_event(self, event):
        if self.start_point is not None:
            self.end_point = event.pos()
            self.image = self.image_copy.copy()
            cv2.rectangle(self.image, (self.start_point.x(), self.start_point.y()),
                          (self.end_point.x(), self.end_point.y()), (0, 255, 0), 2)
            self.update_image(self.image)

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.start_point is not None:
            self.end_point = event.pos()
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()

            # Print the coordinates of the selected region
            print(f"Start Point: (x1={x1}, y1={y1})")
            print(f"End Point: (x2={x2}, y2={y2})")

            self.selected_region = self.image_copy[y1:y2, x1:x2]
            self.update_image(self.image)

    def reset_selection(self):
        self.start_point = None
        self.end_point = None
        self.selected_region = None
        self.image = self.image_copy.copy()
        self.update_image(self.image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    selector = ImageSelector('C:/Users/myash/Downloads/gate.png')
    sys.exit(app.exec_())
