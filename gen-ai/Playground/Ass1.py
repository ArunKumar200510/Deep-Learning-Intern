import cv2

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    top_left = (100, 100)
    bottom_right = (200, 200)

    color = (0, 255, 0)  
    thickness = 2  
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    cv2.imshow('Video with Square', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()