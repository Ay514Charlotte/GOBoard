import cv2
import numpy as np
from matplotlib import pyplot as plt
# input
img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_copy = img_gray.copy()
kernel = np.ones((3, 3), np.uint8)

img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
img_gray = cv2.dilate(img_gray, kernel, iterations=1)
# img_gray = cv2.adaptiveThreshold(img_gray, 255, cv#2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                # cv2.THRESH_BINARY, 11, 2)
ret, img_gray = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
img_gray = cv2.dilate(img_gray, kernel, iterations=1)
# img_gray = cv2.Canny(img_gray, 10, 100)
# img_2 = edges.copy()
# kernel = np.ones((1, 1), np.uint8)
# img_2 = cv2.GaussianBlur(edges, (5, 5), 0)
# img_2 = cv2.Canny(img_2, 100, 200)
# img_2 = cv2.dilate(img_1, kernel, iterations=1)
plt.subplot(1, 1, 1), plt.imshow(img_gray, 'gray')
plt.title('test')
plt.xticks([]), plt.yticks([])
plt.show()


