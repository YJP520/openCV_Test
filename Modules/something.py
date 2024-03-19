##########################################################################
#
#   Project:    Opencv Tests
#   Author :    Yu.J.P
#   Time   :    2024/03/18 -
#
##########################################################################

import os
import cv2 as cv
from PIL import Image
import numpy as np


def getImageAndLabels(path):
    # 存储人脸数据
    faceSamples = []
    # 存储姓名数据
    ids = []
    # 储存图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 人脸检测分类器
    face_detector = cv.CascadeClassifier(
        "D:\PyCharm\python 3.8.5\Lib\site-packages\cv2/data/haarcascade_frontalface_default.xml")
    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片，灰度化
        PIL_img = Image.open(imagePath).convert('L')
        # 把图像转换为数组，
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 预防无面容照片
        for x, y, w, h in faces:
            ids.append(id)
            faceSamples.append(img_numpy[y:y + h, x:x + w])

            # 打印脸部特征和id
        print('id:', id)
        print('fs:', faceSamples)
        return faceSamples, ids


if __name__ == '__main__':
    # 图片路径
    path = '../Train_Test/data/jm/'
    # 获取图像数组和id标签数组
    faces, ids = getImageAndLabels(path)

    # 加载识别器
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write('../Train_Test/trainer/trainer.yml')
