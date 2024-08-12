import cv2
import numpy as np
# img = cv2.imread(r"D:\WhatsApp Image 2024-06-18 at 18.01.48.jpeg")

# height, width = img.shape[:2]

# q_height, q_width = height/4, width / 4
# T = np.float32([[1,0,q_width],[0,1,q_height]])

# img_tran = cv2.warpAffine(img,T,(width,height))

# cv2.imshow("Org",img)
# cv2.imshow("Tran",img_tran)

# cv2.waitKey()
# cv2.destroyAllWindows()

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, imageFrame = webcam.read()
    if not ret:
        break
    
    # Convert the frame to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    
    # Define the range for red color in HSV
    red_lower = np.array([130, 80, 110], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
    # Define the range for green color in HSV
    green_lower = np.array([25, 50, 75], np.uint8)
    green_upper = np.array([85, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    
    # Define the range for blue color in HSV
    blue_lower = np.array([95, 100, 20], np.uint8)
    blue_upper = np.array([125, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    # Apply dilation to the masks
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)
    
    # Find contours for the red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    # Find contours for the green color
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    
    # Find contours for the blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    
    # Display the resulting frame
    cv2.imshow("Color Detection", imageFrame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
