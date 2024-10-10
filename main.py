#import các thư viện và model
from keras.models import load_model
model = load_model("modelEYE_1_better.h5")
import cv2
import numpy as np

path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
#kiểm tra webcam
if not cap.isOpened():
  cap = cv2.VideoCapture(0)
if not cap.isOpened():
  raise IOError("Cannot open webcam")
countDis = 0
countClose = 0
while True:
    alarmThreshold_Sleep = 5
    alarmThreshold_Distraction = 20
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    if len(faces) == 0:
        status = 'Tap trung'
        countDis +=1
        if countDis >= alarmThreshold_Distraction:
            cv2.putText(frame, "phat hien su mat tap trung!",
                        (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,(0, 0, 255),
                        lineType=cv2.LINE_AA)
            cv2.imshow('Distraction', frame)
    else:
        countDis =0
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye_tree_eyeglasses.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray,1.1,4)
            for x,y,w,h in eyes:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                eyess = eye_cascade.detectMultiScale(roi_gray)
                if len(eyess) == 0:
                    print("eyes are not detected")
                    status = "Closed"
                    countClose +=1
                else:
                    for (ex, ey, ew, eh) in eyess:
                        eyes_roi = roi_color[ey:ey+eh, ex:ex + ew]
                        final_image = cv2.resize(eyes_roi, (224,224))
                        final_image = np.expand_dims (final_image, axis =0) # need fourth dimension
                        final_image = final_image/255.0
                        Predictions = model.predict(final_image)

                    if (Predictions>0):
                        status = "Opened"
                        countClose = 0

                if countClose > alarmThreshold_Sleep:
                    cv2.putText(frame, "da phat hien giac ngu", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
                    cv2.imshow('Sleep Detection', frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # sử dụng thư viện putTest() để chèn văn bản vào video
    cv2.putText(frame,status,(100, 100),font,3,(0, 255, 0),2,cv2.LINE_AA)
    cv2.imshow('Drowsiness Detection Tutorial', frame)
    if cv2.waitKey(2) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()