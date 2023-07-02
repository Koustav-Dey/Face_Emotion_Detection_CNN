import cv2
from keras.models import model_from_json
import numpy as np


json_file = open("model.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotion_ditector.h5") # load training model
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(haar_file)# haarcascade_frontalface_default use for face detection

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0


cam = cv2.VideoCapture(0)
labels = {0: 'Angry', 
          1: 'Disgust',
          2: 'Fear',
          3: 'Happy',
          4: 'Neutral',
          5: 'Sad',
          6: 'Surprise'}



while True:
    
    ret , frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)
    
    try:
        for (x,y,w,h) in faces:
            image = gray[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            # cv2.putText(frame,prediction_label)
            cv2.putText(frame, '% s' %(prediction_label), (x-10, y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",frame)
    
        
    except cv2.error:
        cv2.putText(image,"Face Not Found",(220,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow('Face Cropper', image)
        print("Can't Ditect Anything")
        pass
    
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()