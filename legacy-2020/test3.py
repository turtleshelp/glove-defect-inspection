import cv2
import keyboard as keyboard
import numpy as np


def nothing(x):
    pass


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Images/glove4.mp4')
cv2.namedWindow("Trackbars")

cv2.createTrackbar("Low H", "Trackbars", 90, 179, nothing)
cv2.createTrackbar("Low S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Low V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 121, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((15, 15), np.float32) / 225
    smoothed = cv2.filter2D(hsv, -1, kernel)

    l_h = cv2.getTrackbarPos("Low H", "Trackbars")
    l_s = cv2.getTrackbarPos("Low S", "Trackbars")
    l_v = cv2.getTrackbarPos("Low V", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    # lower_blue = np.array([l_h, l_s, l_v])
    # upper_blue = np.array([u_h, u_s, u_v])
    # mask = cv2.inRange(smoothed, lower_blue, upper_blue)
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    lower_blue = np.array([90, 100, 0], dtype=np.uint8)
    upper_blue = np.array([120, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        area = cv2.contourArea(c)
        print("area = ", area)

        x1, y1, w, h = cv2.boundingRect(c)
        rect_area = w * h
        extent = float(area) / rect_area
        print("extent = ", extent)

        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 10

        if 15000 > area and extent < 0.3:

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
            cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
            cv2.putText(frame, "Hole", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        elif area < 1000 and extent > 0.6:

            # cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
            (x1, y1), radius = cv2.minEnclosingCircle(c)
            center = (int(x1), int(y1))
            radius = int(radius) + 20
            cv2.circle(frame, center, radius, (0, 0, 255), 2)
            cv2.putText(frame, "Stain", (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    contours1, hierarchy1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c1 in contours1:
        area1 = cv2.contourArea(c1)
        # print("area1 = ", area1)

        x, y, w, h = cv2.boundingRect(c1)
        rect_area = w * h
        extent = float(area1) / rect_area
        # print("extent = ", extent)

        if area1 < 3000 and extent > 0.5:
            rect = cv2.minAreaRect(c1)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(frame, c1, -1, (0, 255, 0), 3)
            cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
            cv2.putText(frame, "Tear", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    resize_frame = cv2.resize(frame, (600, 400))
    cv2.imshow("Defects", resize_frame)
    resize_mask = cv2.resize(mask, (600, 400))
    cv2.imshow("Mask", resize_mask)
    resize_mask1 = cv2.resize(mask1, (600, 400))
    cv2.imshow("Mask1", resize_mask1)
    resize_blur = cv2.resize(blurred_frame, (600, 400))
    cv2.imshow("blur", resize_blur)
    resize_smoothed = cv2.resize(smoothed, (600, 400))
    cv2.imshow("smooth", resize_smoothed)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()
