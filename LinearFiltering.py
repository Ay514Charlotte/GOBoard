# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
# pic1,pic2,pic3,pic4
# img = cv2.imread('pic2.jpg')
img = cv2.imread('pic1.png')
# img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dst = img_gray.copy()

# 设置kernel
kernel = np.ones((5, 5), np.int16)
kernel[2][2] = -24

# 设置卷积核
kernel2 = np.ones((5, 5), np.int16)
kernel3 = np.ones((3, 3), np.int16)
dst = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel2)

# 平滑
# 高斯
dst = cv2.blur(dst, (5, 5))
# dst = cv2.GaussianBlur(dst, (5, 5), 0)

# 锐化
blur = cv2.Laplacian(dst, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(blur)

# 线性滤波,低通滤波
dst = cv2.filter2D(img_gray, -1, kernel)

# 平滑（双边滤波）
# 双边滤波
dst = cv2.bilateralFilter(dst, 9, 100, 100)

# 在二值化前进行膨胀，以增强线段
dst = cv2.dilate(dst, kernel3)

# 二值化
ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# 显示
plt.subplot(1, 2, 1), plt.imshow(dst, 'gray')
plt.title('test')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(img_gray, 'gray')
plt.title('ori')
plt.xticks([]), plt.yticks([])

plt.show()
