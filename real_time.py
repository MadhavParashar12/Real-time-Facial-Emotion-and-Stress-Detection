import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import json
# import stress_detector.py 

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
    test_image=cap.read()
    test_image=test_image[1]
    converted_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)


    faces_detected = face_haar_cascade.detectMultiScale(converted_image)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_image,(x,y), (x+w,y+h), (255,0,0))
        roi_gray = converted_image[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48))
        image_pixels = image.img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255


        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])

        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]
        label_position = (x,y-10)


        cv2.putText(test_image, emotion_prediction,(50, 50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
        # cv2.putText(test_image,label_position,(50, 50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)


        resize_image = cv2.resize(test_image[1], (1000, 700))
        cv2.imshow('Emotion',resize_image)
        
    else:
            cv2.putText(test_image[1],'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('Emotion Detector',test_image[1])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows