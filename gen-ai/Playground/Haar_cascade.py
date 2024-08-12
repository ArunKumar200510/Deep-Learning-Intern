import cv2
import numpy as np
# Load the haar cascade

# face_det = cv2.CascadeClassifier('gen-ai\Playground\haarcascade_frontalface_default.xml')
# eye_det = cv2.CascadeClassifier('gen-ai\Playground\haarcascade_eye.xml')

# img = cv2.imread("D:\phoro.jpg")
# gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces = face_det.detectMultiScale(gray,1.3,5)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



face_det = cv2.CascadeClassifier('gen-ai/Playground/haarcascade_frontalface_default.xml')
eye_det = cv2.CascadeClassifier('gen-ai/Playground/haarcascade_eye.xml')
smile_det = cv2.CascadeClassifier('gen-ai/Playground/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_det.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), thickness=2)

        smiles = smile_det.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), thickness=2)

    cv2.imshow("Face, Eye, and Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
