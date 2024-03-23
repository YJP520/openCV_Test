########################################################################################################################
#
#   Project:    OpenCV application for the design of graduation.
#   Author :    Yu.J.P
#   Time   :    2024/03/19 -
#
########################################################################################################################

import os
import random
import time

import cv2 as cv
from PIL import Image
import numpy as np

########################################################################################################################


class Camera_Check:
    """ 构造类 """
    def __init__(self):
        # 类变量
        self.camera_value = 0
        self.value = 0
        # 脸部位置
        self.face_x_y = []
        # 读取摄像头 0: 电脑自带, 1: USB
        self.cap = cv.VideoCapture(self.camera_value)

    def get_randon_RGB(self):
        self.value = random.randint(0, 10)
        if self.value == 0:
            return 255, 99, 71
        elif self.value == 1:
            return 255, 140, 0
        elif self.value == 2:
            return 255, 255, 0
        elif self.value == 3:
            return 127, 255, 0
        elif self.value == 4:
            return 0, 255, 0
        elif self.value == 5:
            return 152, 251, 152
        elif self.value == 6:
            return 0, 250, 154
        elif self.value == 7:
            return 0, 255, 255
        elif self.value == 8:
            return 127, 255, 212  # 水上海洋
        elif self.value == 9:
            return 30, 144, 255  # 道奇蓝
        else:
            return 255, 20, 147  # 深粉红色

    """ 图像检测方法 """
    def check_method(self):
        # 设置一个计数器 0 - 250
        count = 0
        while True:
            # 循环初始化
            self.face_x_y = []
            is_check = False
            # 获取标记位、图片
            flag, img = self.cap.read()
            if not flag:
                break
            grey_img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
            # 定义分类器
            face_detector = cv.CascadeClassifier(
                "D:/PyCharm/python 3.8.5/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
            # 使用分类器 获取脸部位置坐标
            face_position = face_detector.detectMultiScale(grey_img, 1.05, 3)
            if str(face_position) != str("()"):
                is_check = True
                count += 1  # 计数器自增
                # 位置标记
                for x, y, w, h in face_position:
                    # 添加脸部坐标
                    self.face_x_y.append((x + w / 2, y + h / 2))
                    # 获取坐标 绘制图形 pt1(x, y), pt2(x+w, y+h)
                    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    # R_G_B = self.get_randon_RGB()
                    R_G_B = 152, 251, 152
                    # cv.rectangle(img, (x, y), (x + w, y + h), R_G_B, 1)
                    # 左上角
                    cv.line(img, (x, y), (x, y + 10), R_G_B, 2)
                    cv.line(img, (x, y), (x + 10, y), R_G_B, 2)
                    # 右上角
                    cv.line(img, (x + w, y), (x + w - 10, y), R_G_B, 2)
                    cv.line(img, (x + w, y), (x + w, y + 10), R_G_B, 2)
                    # 左下角
                    cv.line(img, (x, y + h), (x + 10, y + h), R_G_B, 2)
                    cv.line(img, (x, y + h), (x, y + h - 10), R_G_B, 2)
                    # 右下角
                    cv.line(img, (x + w, y + h), (x + w - 10, y + h), R_G_B, 2)
                    cv.line(img, (x + w, y + h), (x + w, y + h - 10), R_G_B, 2)

            # print('[Debug] - count = ', count)
            if count >= 10 and is_check:
                print('[STATE] - Now I find your face in this screen at', self.face_x_y)
                # print('[STATE] - And count = ', count, ', Time is ', time.time())
                count = 0  # 清零

            cv.imshow("Face-Check", img)
            if ord('q') == cv.waitKey(1):
                break

########################################################################################################################


# 整体测试
if __name__ == '__main__':
    # 测试
    cam = Camera_Check()
    cam.check_method()

















