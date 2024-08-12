import cv2

upper_det = cv2.CascadeClassifier('gen-ai/Playground/haarcascade_upperbody.xml')
lower_det = cv2.CascadeClassifier('gen-ai/Playground/haarcascade_lowerbody.xml')

img = cv2.imread(r"C:\Users\rrak\Downloads\loooo.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
upper_bodies = upper_det.detectMultiScale(gray, 1.1, 5,minSize=(30,30))

for (x, y, w, h) in upper_bodies:
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
    
lower_bodies = lower_det.detectMultiScale(gray,1.3, 5,minSize=(30,30))

for (x, y, w, h) in lower_bodies:
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)


cv2.imshow("Upper and lower",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
