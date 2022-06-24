from flask import Flask,request, render_template, Response, session, redirect
from functools import wraps
from datetime import datetime
import datetime as dt
import pymongo
import cv2
import tensorflow as tf
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from datetime import date
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#import cv2
import numpy as np
import cvlib
import matplotlib.pyplot as plt
import cv2
import face_recognition
import os
import numpy as np
app=Flask(__name__)
camera = cv2.VideoCapture(0)

app = Flask(__name__)
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'


# Database
client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system
path="Customers"

images = []
classNames =[]
mylist = os.listdir(path)
# print(mylist)

# print(cv2.imread(f'{path}/{mylist[0]}'))
for i in range(len(mylist)):
    curImg = cv2.imread(f'{path}/{mylist[i]}')
    images.append(curImg)
    classNames.append(os.path.splitext(mylist[i])[0].split(".")[0])
encodeList = []
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_face = face_recognition.face_encodings(img)[0]
    encodeList.append(encoded_face)
encoded_face_train=encodeList

#facedectation
def gen_frames(address):
    flag=0
    settings=db.settings
    cam_set=settings.find_one({'camera_address':address})
    camera = cv2.VideoCapture(0)
    classifier =load_model('model/Emotion_Detection.h5')
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    if address!='0':
      camera.open(address)
      flag=1
    while True:
        cam_set=settings.find_one({'camera_address':address})
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
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                roi_rgb=rgb[startY-30:endY+30,startX-30:endX+30]
                roi_gray = gray[startY-30:endY+30,startX-30:endX+30]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                roi_rgb=cv2.resize(roi_rgb, (0,0), None, 0.25,0.25)
                if np.sum([roi_gray])!=0:
                  roi = roi_gray.astype('float')/255.0
                  roi = img_to_array(roi)
                  roi = np.expand_dims(roi,axis=0)

              # make a prediction on the ROI, then lookup the class

                  preds = classifier.predict(roi)[0]
                  print(preds)
                  label_position = (startX,startY)
                  label=class_labels[preds.argmax()]
                  cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                if(flag!=1):

                  encoded_face = face_recognition.face_encodings(roi_rgb)[0]

                  matches = face_recognition.compare_faces(encoded_face_train, encoded_face)
                  faceDist = face_recognition.face_distance(encoded_face_train, encoded_face)
                  matchIndex = np.argmin(faceDist)

                if np.sum([roi_gray])!=0:
                  roi = roi_gray.astype('float')/255.0
                  roi = img_to_array(roi)
                  roi = np.expand_dims(roi,axis=0)

              # make a prediction on the ROI, then lookup the class

                  preds = classifier.predict(roi)[0]
                  print(preds)
                  label_position = (startX,startY)
                  label=class_labels[preds.argmax()]
                  cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                  if flag!=1 and matches[matchIndex]:
                    name = classNames[matchIndex].upper().lower()
                    if(cam_set['is_started']=='1'):
                      feed=np.array([preds[1],preds[3]])
                      db.customer_feeds.insert_one({'customer_name':name,'date':date.isoformat(date.today()),'feed':feed.tolist()})
                    cv2.putText(frame,name, (startX,startY-25), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                  if(flag==0 and cam_set['is_started']=='1'):
                    db.records.insert_one({'time':datetime.isoformat(datetime.now()),'cam_address':address,'emotions':preds.tolist()})
                  
                  #print("\nprediction max = ",preds.argmax())
                  # print("\nlabel = ",label)
                  
                  
                  print(label)
                else:
                  cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
          except Exception as e:
            print(e)

          ret, buffer = cv2.imencode('.jpg', frame)
          frame = buffer.tobytes()
          yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def stopp(cam_address):
    db.settings.update_one({'camera_address':cam_address},{'$set':{'is_started':'0'}})
    current_record=db.crecords.find_one({'cam_address':cam_address,'type':'start'})
    print(current_record)
    end=datetime.isoformat(datetime.now())
    db.crecords.delete_one({'cam_address':cam_address,'type':'start'})
    return current_record['time'],end

def enterRecords(start,end,cam_address):
    records=db.records.find({"time":{'$gte': start,'$lte':end}})
    # for i in range(0,27):
    #     print(records[i])
    t=0
    array=[0,0,0,0,0]
    for i in records:
        t+=1
        for j in range(0,5):
            array[j]+=i['emotions'][j]
        #print(i['emotions'])
    for i in range(0,len(array)):
        array[i]/=t
    print(t)
    print(array)
    array=np.array(array)
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    db.history.insert_one({'cam_address':cam_address,'feedback':class_labels[array.argmax()],'from':start,'to':end})

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
def plotpie(start,end):
    emo_dict={'Angry':0,'Happy':1,'Neutral':2,'Sad':3,'Surprise':4}
    start_time=datetime.strptime(start,"%Y-%m-%d")
    end_time=datetime.strptime(end,"%Y-%m-%d")
    delta = dt.timedelta(days=1)
    #end_time+=delta
    happy=[];sad=[];neutral=[];surprise=[];angry=[]
    dates=[]
    totals=[0,0,0,0,0]
    while(start_time<=end_time):
        records=db.history.find({'from':{'$gte': str(start_time),'$lte':str(start_time+delta)}})
        hap=0;sa=0;neut=0;sur=0;ang=0
        for i in records:
            print(i['feedback'])
            if(i['feedback']=='Angry'):
                ang+=1
            elif(i['feedback']=='Happy'):
                hap+=1
            elif(i['feedback']=='Neutral'):
                neut+=1
            elif(i['feedback']=='Sad'):
                sa+=1
            elif(i['feedback']=='Surprise'):
                sur+=1
        happy.append(hap)
        totals[1]+=hap
        sad.append(sa)
        totals[3]+=sa
        neutral.append(neut)
        totals[2]+=neut
        surprise.append(sur)
        totals[4]+=sur
        angry.append(ang)
        totals[0]+=ang
        #print(emotions)
        print(start_time,end="\n")
        dates.append(start_time.date())
        start_time+=delta
    emotions=[0,0,0,0,0]
    #records=db.history.find({'from':{'$gte': start,'$lte':end}})
    #records=db.history.find({'cam_address':"0"})
    print(neutral)
    print(records)
    print("records")
    wp = { 'linewidth' : 1, 'edgecolor' : "green" }
    labels=['Angry','Happy','Neutral','Sad','Surprise']
    colors=('red','green','blue','yellow','purple')
    explode = (0.1, 0.0, 0.2, 0.3, 0.0)
    # Creating plot
    fig, ax = plt.subplots(figsize =(10, 7))
    wedges, texts, autotexts = ax.pie(totals,
                                  autopct = lambda pct: func(pct, totals),
                                  explode=explode,
                                  labels = labels,
                                  shadow = True,
                                  colors=colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))
 
    # Adding legend
    ax.legend(wedges, labels,
          title ="Emotions",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))
 
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Customer Responses")
    # show plot
    plt.title("Customer Responses")
    plt.savefig("static/images/output.jpg")
def analyze(start,end):
    emo_dict={'Angry':0,'Happy':1,'Neutral':2,'Sad':3,'Surprise':4}
    start_time=datetime.strptime(start,"%Y-%m-%d")
    end_time=datetime.strptime(end,"%Y-%m-%d")
    delta = dt.timedelta(days=1)
    happy=[];sad=[];neutral=[];surprise=[];angry=[]
    dates=[]
    totals=[0,0,0,0,0]
    while(start_time<=end_time):
        records=db.history.find({'from':{'$gte': str(start_time),'$lte':str(start_time+delta)}})
        hap=0;sa=0;neut=0;sur=0;ang=0
        for i in records:
            print(i['feedback'])
            if(i['feedback']=='Angry'):
                ang+=1
            elif(i['feedback']=='Happy'):
                hap+=1
            elif(i['feedback']=='Neutral'):
                neut+=1
            elif(i['feedback']=='Sad'):
                sa+=1
            elif(i['feedback']=='Surprise'):
                sur+=1
        happy.append(hap)
        totals[1]+=hap
        sad.append(sa)
        totals[3]+=sa
        neutral.append(neut)
        totals[2]+=neut
        surprise.append(sur)
        totals[4]+=sur
        angry.append(ang)
        totals[0]+=ang
        #print(emotions)
        print(start_time,end="\n")
        dates.append(start_time.date())
        start_time+=delta
    
    return dates,angry,happy,neutral,sad,surprise,totals

def plotHappyIndex(data):
  dates=db.customer_feeds.distinct("date",{'customer_name':data})
  harray=[]
  sarray=[]
  for i in dates:
      date_data=db.customer_feeds.find({'date':i,'customer_name':data})
      total=0
      happy=0
      sad=0
      for j in date_data:
          total+=1
          happy+=j['feed'][0]
          sad+=j['feed'][1]
      happy/=total
      sad/=total
      harray.append(happy)
      sarray.append(sad)
      
  print(harray)
  print(sarray)

  h=np.array(harray)
  s=np.array(sarray)
  plt.plot(h,linestyle='solid',c='green')
  plt.plot(s,linestyle='dotted',c='red')
  plt.title("HappyIndex")
  plt.savefig("static/images/happyindex.jpg")


# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap

# Routes
from user import routes

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/dashboard/')
@login_required
def dashboard():
  cams=db.cameras
  print("Hello " ,cams)
  cameras=[]
  for i in cams.find({}):
    print(i['address'])
    cameras.append(i['address'])
  print("cameras= ",cameras)
  emotions=['Angry','Happy','Neutral','Sad','Surprise']
  evalues=[1,2,3,4,5]
  return render_template('dashboard.html',len=len(cameras),cameras=cameras,emotions=emotions,evalues=evalues,start=0,stop=1)

def index():
    return render_template('dashboard.html')
@app.route('/video_feed',methods=['GET'])
def video_feed():
    args=request.args
    print(args)
    return Response(gen_frames(args.get('address')), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics/')
@login_required
def analytics():
  return render_template('analytics.html')

@app.route('/recents')
def recents():
  records=db.history.find().sort("from",-1)
  return render_template('recents.html',records=records)
@app.route('/start',methods=['POST'])
def start():
  data=request.form.to_dict()
  db.crecords.insert_one({'cam_address':data['start'],'time':datetime.isoformat(datetime.now()),'type':'start'})
  check=db.settings.update_one({'camera_address':'0'},{'$set':{'is_started':'1'}})
  cams=db.cameras
  cameras=[]
  for i in cams.find({}):
    cameras.append(i['address'])
  emotions=['Angry','Happy','Neutral','Sad','Surprise']
  evalues=[1,2,3,4,5]
  return render_template('dashboard.html',len=len(cameras),cameras=cameras,emotions=emotions,evalues=evalues,start=1,stop=0)
  #return render_template('dashboard.html',)

@app.route('/stop',methods=['POST'])
def stop():
  data=request.form.to_dict()
  start_time,end_time=stopp(data['stop'])
  enterRecords(start_time,end_time,data['stop'])
  cams=db.cameras
  cameras=[]
  for i in cams.find({}):
    cameras.append(i['address'])
  emotions=['Angry','Happy','Neutral','Sad','Surprise']
  evalues=[1,2,3,4,5]
  return render_template('dashboard.html',len=len(cameras),cameras=cameras,emotions=emotions,evalues=evalues,start=0,stop=1)

@app.route('/customerFeed')
def showAll():
  data=db.customers.find()
  return render_template('customers.html',data=data)

@app.route('/happyIndex',methods=['POST'])
def showHappy():
  data=request.form.to_dict()
  plotHappyIndex(data['name'])
  message="Dear "+data['name']+",\nWe saw you were not happy with our services and everyone gets a second chance so please give us a second chance to serve you better and more better\n Use FOOD20 promocode to get 20% discount in your next table booking online"
  return render_template('happyindex.html',name=data['customer'].upper(),email=data['email'],message=message)

@app.route('/sendMail',methods=["POST"])
def sendEmail():
  data=request.form.to_dict()
  msg="email sent successfully to "+data['customer_name']
  body=data['message']
  s_address="virtualblog786@gmail.com"
  s_pass="atharva@786"
  r_address=data['email']
  message = MIMEMultipart()
  message['From'] = s_address
  message['To'] = r_address
  message['Subject']="Mail from XYZ RESTAURANT"
  message.attach(MIMEText(body, 'plain'))
  session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
  session.starttls() #enable security
  session.login(s_address, s_pass) #login with mail_id and password
  text = message.as_string()
  session.sendmail(s_address, r_address, text)
  session.quit()
  data=db.customers.find()
  return render_template('customers.html',set=1,message=msg.upper(),data=data)


@app.route('/plotpie',methods=['POST'])
def plot():
  data=request.form.to_dict()
  plotpie(data['start'],data['stop'])
  return render_template('analytics.html',set=1)


@app.route('/history')
def history():
  return render_template('history.html')


@app.route('/analyze',methods=["POST"])
def analyse():
  data=request.form.to_dict()
  dates,angry,happy,neutral,sad,surprise,totals=analyze(data['start'],data['stop'])
  return render_template('history.html',set=1,length=len(dates),dates=dates,angry=angry,happy=happy,neutral=neutral,sad=sad,surprise=surprise)


if __name__=='__main__':
    app.run(debug=True)
  