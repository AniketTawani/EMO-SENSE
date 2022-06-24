

from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cvlib
app=Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frames():
    camera = cv2.VideoCapture(0)
    classifier =load_model('model/Emotion_Detection.h5')
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    # if address!='0':
    #   camera.open(address)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
          #print("hello")
          try:
            labels = []
            faces,c=cvlib.detect_face(frame)
  #     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  #     faces = face_classifier.detectMultiScale(gray,1.3,5)

            for face in faces:
              #   print(face)
                (startX,startY) = face[0],face[1]
                (endX,endY) = face[2],face[3]
                # draw rectangle over face
        #         roi=
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                roi_gray = gray[startY-30:endY+30,startX-30:endX+30]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


                if np.sum([roi_gray])!=0:
                  roi = roi_gray.astype('float')/255.0
                  roi = img_to_array(roi)
                  roi = np.expand_dims(roi,axis=0)

              # make a prediction on the ROI, then lookup the class

                  preds = classifier.predict(roi)[0]
                  #print("\nprediction = ",preds)
                  label=class_labels[preds.argmax()]
                  #print("\nprediction max = ",preds.argmax())
                  # print("\nlabel = ",label)
                  label_position = (startX,startY)
                  cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                else:
                  cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
          except Exception as e:
            print("error")

          ret, buffer = cv2.imencode('.jpg', frame)
          frame = buffer.tobytes()
          yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('dashboard.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)