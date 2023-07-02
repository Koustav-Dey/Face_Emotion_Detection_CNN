import cv2
from keras.models import model_from_json
import numpy as np
import mediapipe as mp

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

labels = {0: 'Angry', 
          1: 'Disgust',
          2: 'Fear',
          3: 'Happy',
          4: 'Neutral',
          5: 'Sad',
          6: 'Surprise'}


# mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(240,160,0))

cam = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:




    while True:
        
        ret , frame = cam.read()
        frame = cv2.flip(frame, 1)
        # if ret:
        # # Get the height and width of the frame
        #     frame_height, frame_width, _ = frame.shape
        #     print("Frame height:", frame_height)
        #     print("Frame width:", frame_width)
        # else:
        #     print("Failed to read the frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(frame, 1.3, 5)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with Face Mesh
        results = face_mesh.process(rgb_frame)
        
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
                
        #     for face_landmarks in results.multi_face_landmarks:
        # # Draw face mesh landmarks
        #         for landmark in face_landmarks.landmark:
        #             x = int(landmark.x * 640)
        #             y = int(landmark.y * 480)
        #             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        #             cv2.imshow("Output",frame)
        
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image= frame,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=drawing_spec ,connection_drawing_spec=drawing_spec)
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