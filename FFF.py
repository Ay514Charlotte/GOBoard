# -*- coding: utf-8 -*-
# 尝试用傅里叶变换对高频进行滤波
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pic2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dst = img_gray.copy()

# 设置kernel
kernel = np.ones((5, 5), np.int16)
kernel[2][2] = -24
kernel2 = np.ones((5, 5), np.int16)
kernel3 = np.ones((3, 3), np.int16)

# 开运算 补充背景
dst = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel2)

# 傅里叶变换
dst = np.fft.fft2(dst)
dst = np.fft.fftshift(dst)

# 构建振幅
magnitude_spectrum = 20*np.log(np.abs(dst))

plt.subplot(1, 2, 1), plt.imshow(img_gray, 'gray')
plt.title('ori')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(magnitude_spectrum, 'gray')
plt.title('mag')
plt.xticks([]), plt.yticks([])
plt.show()
