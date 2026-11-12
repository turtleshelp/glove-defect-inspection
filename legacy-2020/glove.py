import numpy as np
import cv2

img = cv2.imread("Images/glove.jpeg")
#cv2.imshow('orignal', im)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours = ", str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, 7, (0, 255, 0), 3)

cnt = contours[0]
M = cv2.moments(cnt)
print("M", M)

area = cv2.contourArea(cnt)
print("area = ", area)

perimeter = cv2.arcLength(cnt, True)
print("perimeter = ", perimeter)

x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
aspect_ratio = float(w) / h
extent = float(area) / rect_area
print("aspect ratio = ", aspect_ratio)
print("extent = ", extent)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
#img2 = cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
#cv2.imshow('box', img2)

#hull = cv2.convexHull(cnt)
#hull_area = cv2.contourArea(hull)
#solidity = float(area) / hull_area
#print("solidity = ", solidity)

equi_diameter = np.sqrt(4 * area / np.pi)
print("eq diameter = ", equi_diameter)

#(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
#print("MA =", MA)
#print("ma =", ma)
#print("angle =", angle)


final = cv2.resize(img, (600,600))
cv2.imshow('results', final)
cv2.waitKey(0)
