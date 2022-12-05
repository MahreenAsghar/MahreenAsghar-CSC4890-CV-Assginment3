import numpy as np
import cv2
import os
if not os.path.exists('images'):
    os.makedirs('images')

faceData = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceId = input('\n Enter User ID ')
print("\n Capturing ")


while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceData.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("./images/Users." + str(faceId) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k < 30:
        break
    elif count >= 30:
         break

print("\n Code run ended")
cam.release()
cv2.destroyAllWindows()
