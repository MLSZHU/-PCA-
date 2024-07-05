import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt,QSize,QRect
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPixmap,QImage,QFont
from PyQt5.QtCore import QTimer
from utils import rec_face
import random

name = {
    1:"蒋卓璞",
    2:"林远福", 
    3:"彭瑞峰", 
    4:"石祥金", 
    5:"吴萧", 
    6:"俞天宸", 
    7:"张家喻", 
    8:"朱亦晨",
}

w = 640
h = 480
class FaceInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建布局
        layout = QVBoxLayout()

        # 创建两个QLabel
        self.faceLabel = QLabel(self)
        self.faceLabel.setFixedSize(h,h)   # 设置人脸图像的大小
        self.nameLabel = QLabel(self)
        self.nameLabel.setAlignment(Qt.AlignCenter)   # 设置人名标签的对齐方式
        self.name = None

        # 创建四个按钮
        self.captureButton = QPushButton('拍摄', self)
        self.recognizeButton = QPushButton('识别', self)
        self.obscureButton = QPushButton('随机遮挡', self)
        self.reconstructButton = QPushButton('重构', self)

        # 添加组件到布局
        layout.addWidget(self.faceLabel)
        layout.addWidget(self.nameLabel)
        layout.addWidget(self.captureButton)
        layout.addWidget(self.recognizeButton)
        layout.addWidget(self.obscureButton)
        layout.addWidget(self.reconstructButton)

        # 设置布局
        self.setLayout(layout)
        self.frame = None

        # 连接按钮的点击事件
        self.captureButton.clicked.connect(self.captureImage)
        self.recognizeButton.clicked.connect(self.recognizeFace)
        self.obscureButton.clicked.connect(self.obscureImage)
        self.reconstructButton.clicked.connect(self.reconstructImage)

        # 设置窗口
        self.setWindowTitle('人脸识别与重构')      
        self.setGeometry(800, 400, h, h)

    def openCamera(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        # 创建定时器，用于定时更新显示的图像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateCameraImage)
        self.timer.start(30)  # 每30毫秒更新一次图像

    def updateCameraImage(self):
        # 从摄像头读取图像
        ret, self.frame = self.cap.read()
        # print(self.frame.shape)
        self.frame = cv2.resize(self.frame, (h, w))
        if ret:
            # 将图像转换为QImage
            qimg = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_BGR888)
            # 将QImage转换为QPixmap
            qimg_pixmap = QPixmap.fromImage(qimg,Qt.AutoColor)

            self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))


    def captureImage(self):
        self.openCamera()

    def recognizeFace(self):
        ret, self.frame = self.cap.read()
        # print(type(frame))
        # if ret:
        self.frame = cv2.imread(r"E:\Desktop\tp3(1)\tp3\validation\1.JPG")
        self.frame = cv2.resize(self.frame, (h, w))
        #     # 将图像转换为QImage
        if not np.any(self.frame):
            image = cv2.imread("./tishi.jpg")
            qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_BGR888)
            # 将QImage转换为QPixmap
            qimg_pixmap = QPixmap.fromImage(qimg, Qt.AutoColor)
            # 显示图像
            self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))
            return
        print(self.frame.shape)
        qimg = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_BGR888)
        # 将QImage转换为QPixmap
        qimg_pixmap = QPixmap.fromImage(qimg, Qt.AutoColor)
        # 显示图像
        self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))
            # 关闭摄像头
        self.cap.release()
        # 停止定时器
        self.timer.stop()
        
        person_id, _ = rec_face(self.frame)
        self.nameLabel.setText(name[person_id])
        self.nameLabel.setFont(QFont("Arial", 18))
            # print(person_id)

    def obscureImage(self):
        if not np.any(self.frame):
            image = cv2.imread("./tishi.jpg")
            qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_BGR888)
            # 将QImage转换为QPixmap
            qimg_pixmap = QPixmap.fromImage(qimg, Qt.AutoColor)
            # 显示图像
            self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))
            return
        image = self.frame
        # 设置遮挡区域的大小范围
        min_width = 20
        max_width = 50
        min_height = 20
        max_height = 50
        # 随机生成遮挡区域的大小
        width = np.random.randint(min_width, max_width)
        height = np.random.randint(min_height, max_height)
        # 随机生成遮挡区域的起始位置
        x = np.random.randint(0, image.shape[1] - width)
        y = np.random.randint(0, image.shape[0] - height)
        # 遮挡图像
        image[y:y+height, x:x+width] = 0  # 将遮挡区域设置为黑色（0, 0, 0）
        self.frame = image
        qimg = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_BGR888)
        # 将QImage转换为QPixmap
        qimg_pixmap = QPixmap.fromImage(qimg, Qt.AutoColor)
        # 显示图像
        self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))
        self.cap.release()
        # 停止定时器
        self.timer.stop()

    def reconstructImage(self):
        if not np.any(self.frame):
            image = cv2.imread("./tishi.jpg")
            qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_BGR888)
            # 将QImage转换为QPixmap
            qimg_pixmap = QPixmap.fromImage(qimg, Qt.AutoColor)
            # 显示图像
            self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))
            return
        # 这里应该是图像重构的代码
        _, reconstructed_image = rec_face(self.frame)
        import matplotlib.pyplot as plt
        img = reconstructed_image.reshape(640, 480)
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
        print(self.frame.shape)
        img = reconstructed_image.reshape(w,h)
        # img = img.astype(np.uint8)
        # print(img.shape)
        qimg = QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale16)
        # 将QImage转换为QPixmap
        qimg_pixmap = QPixmap.fromImage(qimg, Qt.AutoColor)
        # 显示图像
        self.faceLabel.setPixmap(qimg_pixmap.scaled(QSize(self.faceLabel.size().width(),self.faceLabel.size().height()), Qt.KeepAspectRatio))
            # 关闭摄像头
        self.cap.release()
        # 停止定时器
        self.timer.stop()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceInterface()
    ex.show()
    sys.exit(app.exec_())