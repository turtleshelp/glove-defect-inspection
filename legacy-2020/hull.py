import numpy as np
import cv2

hand = cv2.imread("Images/glove.jpeg", 0)

ret, threshold = cv2.threshold(hand, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#basic_contours = cv2.drawContours(hand, contours, -1, (0, 0, 0), 3)

for c in contours:
    area = cv2.contourArea(c)
    print("area = ", area)

    if 200000 > area > 10000:
        hull = [cv2.convexHull(c)]
        final = cv2.drawContours(hand, hull, -1, (0, 0, 255), 3)

#cv2.imshow('Original', hand)
#cv2.imshow('Thresh', threshold)
cv2.imshow('Convex Hull', hand)
cv2.waitKey(0)
