import cv2
import numpy as np
from matplotlib import pyplot as plt
# input
img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gradX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, 5)
gradY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, 5)
img_gray = cv2.add(gradX, gradY)
img_gray = cv2.convertScaleAbs(img_gray)
edges = cv2.GaussianBlur(img_gray, (3, 3), 0)
# edges = cv2.Canny(img_gray, 100, 180)
img_2 = edges.copy()
kernel = np.ones((1, 1), np.uint8)
# img_2 = cv2.GaussianBlur(edges, (5, 5), 0)
# img_2 = cv2.Canny(img_2, 100, 200)
img_2 = cv2.dilate(img_2, kernel, iterations=1)
plt.subplot(1, 1, 1), plt.imshow(img_2, 'gray')
plt.title('test')
plt.xticks([]), plt.yticks([])
plt.show()


