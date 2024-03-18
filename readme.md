# ***一些简单的学习记录***

---

### 1. 读取图片
- cv2.imshow()
  - 展示图像,参数为窗口名,argument 'mat'
  - 参数为图片地址

- cv2.waitKey()
  - 不断刷新图像，频率为delay，单位为ms
  - 返回值为当前键盘按键值
  - 没有按下按键则继续等待
  - 控制imshow()的持续时间

```python
import cv2 as cv            # 导入模块
img = cv.imread('test.png') # 读取图片
cv.imshow("show", img)      # 显示图片
cv.waitKey(0)               # 等待按键
cv.destroyAllWindows()      # 释放内存
```

```python
import cv2 as cv                        # 导入模块
img = cv.imread('../Data/firefly.png')  # 读取图片

new_width = 711
new_height = 400
resize_img = cv.resize(img, (new_width, new_height),
                       interpolation=cv.INTER_LINEAR)

cv.imshow("show-resize-img", resize_img)   # 显示图片
cv.waitKey(0)                       # 等待按键
cv.destroyAllWindows()              # 释放内存
```

#### 测试结果
![avatar](./Images_md/img_resize_show.png)

### 2. 图片灰度化

- 图像灰度化目的：简化矩阵，提高运算速度。
- 彩色图像中的每个像素颜色由R、G、B三个分量来决定，而每个分量的取值范围都在0-255之间，这样对计算机来说，彩色图像的一个像素点就会有256\*256\*256=16777216种颜色的变化范围！而灰度图像是R、G、B分量相同的一种特殊彩色图像，对计算机来说，一个像素点的变化范围只有0-255这256种。彩色图片的信息含量过大，而进行图片识别时，其实只需要使用灰度图像里的信息就足够了，所以图像灰度化的目的就是为了提高运算速度。

---

- cv2.cvtColor()
  - 灰度转换函数
- cv2.imwrite()
  - 保存图片

```python
import cv2 as cv                        # 导入模块
img = cv.imread('../Data/firefly.png')  # 读取图片

new_width, new_height = 711, 400        # 设置新宽长
resize_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
cv.imshow("Show-Resize-Img", resize_img)        # 显示图片

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度转换
cv.imwrite("../Data/gray_img.png", gray_img)    # 写入图片

resize_gray_img = cv.resize(gray_img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
cv.imshow("Gray-Img", resize_gray_img)          # 显示图片


cv.waitKey(0)           # 等待按键
cv.destroyAllWindows()  # 释放内存
```
#### 测试结果
![avatar](./Images_md/img_gray_test.png)

### 尺寸转换

- cv2.resize()
  - 调整图片大小，接收宽长元组

### 输出图片大小

- img.shape

### 按键退出

- ord('q')，返回q的ASCLL码

```python
import cv2 as cv                        # 导入模块
img = cv.imread('../Data/firefly.png')  # 读取图片

new_width, new_height = 711, 400        # 设置新宽长
resize_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
cv.imshow("Show-Resize-Img", resize_img)        # 显示图片

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度转换
cv.imwrite("../Data/gray_img.png", gray_img)    # 写入图片

resize_gray_img = cv.resize(gray_img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
cv.imshow("Gray-Img", resize_gray_img)          # 显示图片

while True:  # 按下 q 时退出程序
    if ord('q') == cv.waitKey(1):
        break
cv.destroyAllWindows()  # 释放内存
```

### 绘制矩形和圆形框

- cv2.rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
  - img 在图片上绘制
  - pt1 矩形的一个顶点
  - pt2 与pt1在对角线上相对的矩形的顶点
  - color 指定边框的颜色，由（B,G,R）组成
  - thinkness 为正值时代表线条粗细，为负值时边框实线

- cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
  - img 输入的图片data
  - center 圆心位置
  - radius 圆的半径
  - color 圆的颜色
  - thickness 圆形轮廓的粗细（如果为正）。负厚度表示要绘制实心圆。

#### 测试结果
![avatar](./Images_md/img_rec_cir.png)

### 人脸检测

在路径 (Lib/site-packages/cv2/data) 下，可以找到需要xml文件，这些是OpenCV中自带的分类器，根据文件名我们可以看到有识别眼睛、身体、脸等等。

```
haarcascade_eye.xml
haarcascade_eye_tree_eyeglasses.xml
haarcascade_frontalface_alt.xml
haarcascade_frontalface_alt_tree.xml
haarcascade_frontalface_alt2.xml
haarcascade_frontalface_default.xml
haarcascade_fullbody.xml
haarcascade_lefteye_2splits.xml
haarcascade_lowerbody.xml
haarcascade_mcs_eyepair_big.xml
haarcascade_mcs_eyepair_small.xml
haarcascade_mcs_leftear.xml
haarcascade_mcs_lefteye.xml
haarcascade_mcs_mouth.xml
haarcascade_mcs_nose.xml
haarcascade_mcs_rightear.xml
haarcascade_mcs_righteye.xml
haarcascade_mcs_upperbody.xml
haarcascade_profileface.xml
haarcascade_righteye_2splits.xml
haarcascade_smile.xml
haarcascade_upperbody.xml
```

- 使用cv.CascadeClassifier(参数：分类器所在路径)方法定义一个分类器对象。

- detectMultiScale(self,
  - image: Any,
  - scaleFactor: Any = None,
  - minNeighbors: Any = None,
  - flags: Any = None,
  - minSize: Any = None,
  - maxSize: Any = None) 

```python
import cv2 as cv  # 导入模块

img = cv.imread('../Data/MandySa.jpg')  # 读取图片
# img = cv.imread('../Data/firefly.png')  # 读取图片

new_width, new_height = 512, 385        # 设置新宽长
# new_width, new_height = 711, 400        # 设置新宽长

resize_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
cv.imshow("Show-Resize-Img", resize_img)  # 显示图片

# 图片灰度化
grey_img = cv.cvtColor(resize_img, cv.COLOR_BGRA2GRAY)
# 定义分类器，使用OpenCV自带的分类器
face_detector = cv.CascadeClassifier(
    'D:\PyCharm\python 3.8.5\Lib\site-packages\cv2/data/haarcascade_frontalface_alt2.xml')
# 使用分类器
face = face_detector.detectMultiScale(grey_img)
print(face)

# 在图片中对人脸画矩阵
for x, y, w, h in face:
    cv.rectangle(resize_img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
cv.imshow('result', resize_img)

while True:  # 按下 q 时退出程序
    if ord('q') == cv.waitKey(1):
        break
cv.destroyAllWindows()  # 释放内存
```

#### 测试结果
![avatar](./Images_md/img_face_check.png)

### 检测多个人脸

```python
import cv2 as cv
 
def face_detect_methed():
    # 图片灰度化
    grey_img = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
    # 定义分类器，使用OpenCV自带的分类器
    face_detector = cv.CascadeClassifier('G:/conda/envs/testOpencv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    # 使用分类器
    face = face_detector.detectMultiScale(grey_img,1.1,5,0,(10,10),(200,200))
    # 在图片中对人脸画矩阵
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)
 
#读取图像
img = cv.imread("faceMorePeople.png")
#调用检测函数
face_detect_methed()
 
while True:
    if ord('m') == cv.waitKey(0):
        break
 
cv.destroyAllWindows()
```

### 对视频的检测

- ***cap = cv2.VideoCapture(filepath)***
  - cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如cv.VideoCapture(0)为电脑自带摄像头，1为外接摄像头。
  - 如果是视频文件，直接指定好路径即可,如 cv.VideoCapture('G:/1.mp4')，即读取在G盘中名为1的MP4视频文件。

- ***flag, frame = cap.read()***
  - 读取视频帧函数
  - 第一个参数flag为True或者False,代表有没有读取到图片
  - 第二个参数frame表示截取到一帧的图片

- ***使用一个循环判断是否捕获到图像***
  - 如果flag==false，说明视频结束，退出循环
  - 否则则继续将视频中捕获到的帧图像放入检测函数face_detect_method中进行检测。

```python
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_method(frame)
    if ord('c')==cv.waitKey(1):
        break
```

- 释放图像 cap.release()
  - 使用结束后释放摄像头资源。

- WaitKey方法
  - 需要设置WaitKey方法的参数为1，如果为0的话则只能捕获到视频的第一帧，不能播放视频。

```python
import cv2 as cv  # 导入模块

# 读取摄像头
cap = cv.VideoCapture(0)

# 读取视频
# cap = cv.VideoCapture('G:/1.mp4')


# 检测方法定义
def face_detect_method(img):
    grey_img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    face_detector = cv.CascadeClassifier(
        "D:\PyCharm\python 3.8.5\Lib\site-packages\cv2/data/haarcascade_frontalface_default.xml")
    face = face_detector.detectMultiScale(grey_img, 1.05, 4)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
    cv.imshow("result", img)


# 循环判断
while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_method(frame)
    if ord('q') == cv.waitKey(1):
        break

cv.destroyAllWindows()
cap.release()
```

### 视频捕获


















