# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 开操作补背景
kernel = np.ones((7, 7), np.uint8)
blur = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

# 尝试采用双边滤波平滑
blur = cv2.bilateralFilter(blur, 9, 100, 100)

# 锐化
blur = cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3)
blur = cv2.convertScaleAbs(blur)

# 闭运算
# kernel2 = np.ones((3, 3), np.uint8)
# blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel2)

# 均值滤波
blur = cv2.blur(blur, (3, 3))

# 二值化
# ret, blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY, 11, 2)

# 开运算
kernel2 = np.ones((1, 1), np.uint8)
blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel2)
# 边缘

# 高斯平滑
# blur = cv2.GaussianBlur(blur, (5, 5), 0)

# 二值化

# 显示图片
plt.subplot(1, 2, 1), plt.imshow(blur, 'gray')
plt.title('test')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(img_gray, 'gray')
plt.title('ori')
plt.xticks([]), plt.yticks([])

plt.show()
