import cv2

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml")

l1 = "Face Detected"
l2 = "Face Not Detected"

cam = cv2.VideoCapture(0)
while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    if len(face)==0:
        cv2.putText(img,l2,(10,20),cv2.FONT_ITALIC, 0.5, (0,0,255),2)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
        cv2.putText(img,l1,(10,20),cv2.FONT_ITALIC, 0.5, (0,0,255),2)
    cv2.imshow("Facedetection",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
