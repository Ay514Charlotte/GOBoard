# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
# pic1,pic2,pic3,pic4
img = cv2.imread('pic1.png')
# img = cv2.imread('pic5.png')
# img = cv2.imread('test.jpg')
img_shape = img.shape
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dst = img_gray.copy()

# 获取图像的长宽
length = img_shape[0]
height = img_shape[1]
midlength = length / 2
midheight = height / 2
quarterL = length / 3
quarterT = height / 3
sigmaX = 2*(quarterL**2)
sigmaY = 2*(quarterT**2)

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
# 膨胀导致线段筛选困难
# dst = cv2.dilate(dst, kernel3)

# 二值化
ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Hough变换前的高斯赋值
# 从图像中央到边缘，逐渐降低权重
for i in range(length):
    for j in range(height):
        dst[i][j] = dst[i][j] * math.exp((i-midlength)**2/sigmaX +
                                         (j-midheight)**2/sigmaY)

# Hough变换
lines = cv2.HoughLines(dst, 1, np.pi/180, 250)
# 利用方差进行平行线的判断
# 近邻生长法
# Hough变换
# 首先找到水平的范围，上边界为rho最大，theta最小，下边界为rho最小，theta最大（<90)
ThetaMIN2 = 180
ThetaMIN = 180
ThetaMAX = -1
ThetaMAX2 = -1
RhoMAX = -1
RhoMAX2 = -1
RhoMIN = 1000
RhoMIN2 = 1000
up = 0
bot = 0
up2 = 0
bot2 = 0

for i in range(lines[0].size/2):
    # 遍历全部点集
    r, t = lines[0][i]
    print(lines[0][i])
    t = t / np.pi*180
    if t < 90:
        # theta < 90，属于水平范围
        if t > ThetaMAX:
            ThetaMAX = t
            up = i
        if r > RhoMAX:
            RhoMAX = r
            bot = i
    else:
        if t > ThetaMAX2:
            ThetaMAX2 = t
            up2 = i
        if t < ThetaMIN2:
            ThetaMIN2 = t
            bot2 = i

result = [up, bot, up2, bot2]
print(result)
# Hough变换
xtheta = []
yrho = []
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    # if theta < 0.5:
    xtheta.append(theta / np.pi * 180)
    yrho.append(rho)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

print(result)
for i in result:
    rho = lines[0][i][0]
    theta = lines[0][i][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)

# 计算最有可能的两条best

# 显示
# plt.subplot(1, 2, 1), plt.imshow(dst, 'gray')

plt.subplot(1, 3, 1), plt.imshow(dst, 'gray')
plt.title('test')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(img_gray, 'gray')
plt.title('ori')
plt.xticks([]), plt.yticks([])

# plt.subplot(1, 4, 3), plt.scatter(xtheta, yrho, c='b', marker='o')
# plt.title('Hough')
# plt.xlabel('X'), plt.ylabel('Y')

plt.subplot(1, 3, 3), plt.imshow(img)
plt.title('Lines')
plt.xticks([]), plt.yticks([])

plt.show()

