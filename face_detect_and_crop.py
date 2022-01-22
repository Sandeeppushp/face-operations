import os
import time
import cv2

cascadePath = "haarcascade_frontalface_default.xml"
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier(cascadePath);

img = cv2.imread('a.jpg')

def run(validate_with_eyes):
    #faces = faceCascade.detectMultiScale(img)
    faces = faceCascade.detectMultiScale(img, 1.3, 15)
    
    if validate_with_eyes==True:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        found=0
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes)==2:
                #print('eyes found')
                cv2.imwrite("12.jpg", img[y-20:y+h, x:x+w])

        if found==0:
            print('eyes found')
        else:
            print('eyes does not found')
        

    else:
        faces = faceCascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            cv2.imwrite("12.jpg", img[y-20:y+h, x:x+w])
        #cv2.imshow('frame',frame)
        #key = cv2.waitKey(1) & 0xFF
        


run(validate_with_eyes=True)
#run(validate_with_eyes=False)
