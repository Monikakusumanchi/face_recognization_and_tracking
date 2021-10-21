import cv2
haar_file = 'haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(haar_file)    #importing casecade algorithm
cam = cv2.VideoCapture(0)
count = 1
while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.3, 4)#
    for (x,y,w,h) in faces:                           #getting facial location
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                                      #draw rectangle
    cv2.imshow('FaceDetection', img)
    key = cv2.waitKey(10)
    if key == 27:   #27 means escape button
         break
cam.release()
cv2.destroyAllWindows()
