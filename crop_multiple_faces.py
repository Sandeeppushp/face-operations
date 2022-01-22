import cv2
import numpy as np

# Load some pre-trained data on face frontal from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('a.jpg')

# Must convert to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

img_crop = []

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_crop.append(img[y:y + h, x:x + w])

for counter, cropped in enumerate(img_crop):
    #cv2.imshow('Cropped', cropped)
    cv2.imwrite("pose_result_{}.png".format(counter), cropped)
    cv2.waitKey(0)
