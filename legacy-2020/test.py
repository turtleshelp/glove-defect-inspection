import cv2
import numpy as np


def nothing(x):
    pass


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Images/vid1.mp4')
frame_counter = 0
cv2.namedWindow("Trackbars")

cv2.createTrackbar("Low H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Low S", "Trackbars", 20, 255, nothing)
cv2.createTrackbar("Low V", "Trackbars", 70, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 20, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    frame_counter = 1
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bilateral_frame = cv2.bilateralFilter(frame, 9, 100, 100)
    # blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(bilateral_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("Low H", "Trackbars")
    l_s = cv2.getTrackbarPos("Low S", "Trackbars")
    l_v = cv2.getTrackbarPos("Low V", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    # lower_blue = np.array([l_h, l_s, l_v])
    # upper_blue = np.array([u_h, u_s, u_v])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    lower_blue = np.array([90, 100, 0], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # lower_skin = np.array([90, 100, 0], dtype=np.uint8)
    # upper_skin = np.array([130, 255, 255], dtype=np.uint8)
    # mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # extrapolate the hand to fill dark spots within
    # mask = cv2.dilate(mask, kernel, iterations=4)

    # blur the image
    # mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        print("area = ", area)

        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 1

        if 200000 > area:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
            cv2.drawContours(frame, c, -1, (0, 255, 0), 3)
            cv2.putText(frame, "Missing", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    resize_frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Frame", resize_frame)
    resize_mask = cv2.resize(mask, (800, 600))
    cv2.imshow("Mask", resize_mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()
