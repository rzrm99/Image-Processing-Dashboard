

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QWidget, QLineEdit, QGridLayout,
    QSplitter, QFormLayout, QHBoxLayout, QSlider, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO


def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)


def add_noise(image, noise_type='gaussian'):
    row, col, ch = image.shape
    if noise_type == "gaussian":
        mean = 0
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
        noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.02
        noisy = image.copy()
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[tuple(coords)] = 255
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[tuple(coords)] = 0
        return noisy
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return image


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Image Processing Dashboard- Reza Ramezani")
        self.setGeometry(100, 100, 1200, 700)

        self.image = None
        self.processed_image = None

        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(512, 384)

        # ==== Controls ====
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        self.color_mode = QComboBox()
        self.color_mode.addItems(["Original", "Grayscale", "RGB"])
        self.color_mode.currentIndexChanged.connect(self.apply_processing)

        self.noise_type = QComboBox()
        self.noise_type.addItems(["None", "Gaussian", "Salt & Pepper", "Poisson"])
        self.noise_type.currentIndexChanged.connect(self.apply_processing)

        self.filter_type = QComboBox()
        self.filter_type.addItems([
            "None", "Blur", "GaussianBlur", "Median", "Bilateral",
            "Edge Detection", "Sharpen", "Custom"
        ])
        self.filter_type.currentIndexChanged.connect(self.apply_processing)

        self.kernel_grid = [[QLineEdit() for _ in range(3)] for _ in range(3)]
        self.kernel_layout = QGridLayout()
        for i in range(3):
            for j in range(3):
                self.kernel_grid[i][j].setFixedWidth(40)
                self.kernel_grid[i][j].setText("0")
                self.kernel_layout.addWidget(self.kernel_grid[i][j], i, j)

        self.run_custom_button = QPushButton("Run Custom Filter")
        self.run_custom_button.clicked.connect(self.apply_custom_kernel)

        self.kernel_example_dropdown = QComboBox()
        self.kernel_example_dropdown.addItems([
            "Select Example...", "Sharpen", "Edge Detection", "Emboss"
        ])
        self.kernel_example_dropdown.currentIndexChanged.connect(self.insert_kernel_example)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_brightness_contrast)
        self.brightness_value = QLabel("0")

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.update_brightness_contrast)
        self.contrast_value = QLabel("0")

        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self.brightness_slider)
        brightness_row.addWidget(self.brightness_value)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self.contrast_slider)
        contrast_row.addWidget(self.contrast_value)

        self.hist_label = QLabel()
        self.hist_label.setFixedSize(256, 200)
        self.hist_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.hist_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hist_label.setAlignment(Qt.AlignCenter)

        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_image)

        # === Left Control Panel Layout ===
        control_panel = QWidget()
        control_layout = QFormLayout()
        control_layout.addRow(self.load_button)
        control_layout.addRow("Color Mode:", self.color_mode)
        control_layout.addRow("Noise:", self.noise_type)
        control_layout.addRow("Filter:", self.filter_type)
        control_layout.addRow("Custom Kernel (3x3):", self.kernel_layout)
        control_layout.addRow(self.run_custom_button)
        control_layout.addRow("Insert Example:", self.kernel_example_dropdown)
        control_layout.addRow(self.reset_button)
        control_layout.addRow("Brightness:", brightness_row)
        control_layout.addRow("Contrast:", contrast_row)
        control_layout.addRow("Histogram:", QLabel())  # Just keeps the label
        control_layout.addRow(self.hist_label)
        control_layout.addRow(self.save_button)

        control_panel.setLayout(control_layout)

        # === Right Panel ===
        image_panel = QWidget()
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_panel.setLayout(image_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(image_panel)
        splitter.setStretchFactor(1, 3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File")
        if file_path:
            self.image = cv2.imread(file_path)
            self.processed_image = self.image.copy()
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.update_display()

    def apply_processing(self):
        if self.image is None:
            return
        img = self.image.copy()

        mode = self.color_mode.currentText()
        if mode == "Grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        noise = self.noise_type.currentText().lower().replace(" ", "_")
        if noise != "none":
            img = add_noise(img, noise)

        filter_type = self.filter_type.currentText()
        if filter_type == "Blur":
            img = cv2.blur(img, (3, 3))
        elif filter_type == "GaussianBlur":
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif filter_type == "Median":
            img = cv2.medianBlur(img, 5)
        elif filter_type == "Bilateral":
            img = cv2.bilateralFilter(img, 9, 75, 75)
        elif filter_type == "Edge Detection":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
        elif filter_type == "Sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

        self.processed_image = img
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.update_display()

    def apply_custom_kernel(self):
        if self.image is None:
            return
        try:
            kernel = np.array([[float(self.kernel_grid[i][j].text()) for j in range(3)] for i in range(3)])
            img = self.processed_image.copy()
            img = cv2.filter2D(img, -1, kernel)
            self.processed_image = img
            self.update_display()
        except Exception as e:
            print("Error applying custom kernel:", e)

    def insert_kernel_example(self):
        choice = self.kernel_example_dropdown.currentText()
        examples = {
            "Sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            "Edge Detection": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            "Emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
        }
        if choice in examples:
            matrix = examples[choice]
            for i in range(3):
                for j in range(3):
                    self.kernel_grid[i][j].setText(str(matrix[i][j]))

    def reset_image(self):
        if self.image is not None:
            self.processed_image = self.image.copy()
            self.color_mode.setCurrentIndex(0)
            self.noise_type.setCurrentIndex(0)
            self.filter_type.setCurrentIndex(0)
            self.kernel_example_dropdown.setCurrentIndex(0)
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            for row in self.kernel_grid:
                for cell in row:
                    cell.setText("0")
            self.update_display()

    def update_brightness_contrast(self):
        if self.image is None:
            return
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        self.brightness_value.setText(str(brightness))
        self.contrast_value.setText(str(contrast))

        img = self.processed_image.copy().astype(np.float32)
        img = img * (contrast / 50.0 + 1) - contrast + brightness
        img = np.clip(img, 0, 255).astype(np.uint8)

        self.update_display(img)

    def update_display(self, img_override=None):
        img = img_override if img_override is not None else self.processed_image
        if img is not None:
            pixmap = convert_cv_qt(img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.update_histogram(img)

    def update_histogram(self, image):
        fig, ax = plt.subplots(figsize=(2.56, 2.0), dpi=100)
        ax.clear()
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_yticks([])
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        pixmap = QPixmap.fromImage(QImage.fromData(buf.read()))
        self.hist_label.setPixmap(pixmap.scaled(
            self.hist_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        plt.close(fig)

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                cv2.imwrite(file_path, self.processed_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageProcessor()
    win.show()
    sys.exit(app.exec_())
