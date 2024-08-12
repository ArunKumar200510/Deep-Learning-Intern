import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([130, 80, 110], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    green_lower = np.array([25, 50, 75], np.uint8)
    green_upper = np.array([110, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    blue_lower = np.array([75, 100, 10], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    purple_lower = np.array([130, 50, 70], np.uint8)
    purple_upper = np.array([160, 255, 255], np.uint8)
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)

    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, purple_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video with Detected Colors', frame)

    if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty('Video with Detected Colors', cv2.WND_PROP_VISIBLE) < 1:
        break

webcam.release()
cv2.destroyAllWindows()