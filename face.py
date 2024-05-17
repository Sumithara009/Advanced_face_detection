import cv2 as cv

name = "haarcascade_frontalface_default.xml"

haar_cascade = cv.CascadeClassifier(name)

cam = cv.VideoCapture(0)#it access primary camera

while True:
    a,img = cam.read()#variable a is only shown whether the img is present or not
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)#convert into gray color image
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv.rectangle(img,(x,y),(x+w,y+h), (255,255,0),5)#draw rectangle
    cv.imshow("FaceDetection",img)#display image
    key = cv.waitKey(10)#time delay
    if key == 27:#it breaks the program
        break
cam.release()
cv.destroyAllWindows()
