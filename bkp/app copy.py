import cv2
import os
from flask import Flask,request,render_template,session,redirect,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_wtf import FlaskForm
from wtforms import (StringField, IntegerField,DateTimeField,SubmitField)
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate



app = Flask(__name__)

face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
app.config['SECRET_KEY'] = 'mysecretkey'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db = SQLAlchemy(app)
#Migrate(db,app)

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

#################################### Landing ######################
@app.route('/')
def home():   
    return render_template('landing.html') 

########################################## CLASS INFORMATION #############################


class ClassInformationForm(FlaskForm):
    number = StringField('Class Number: ',validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/classinformation', methods=["GET", "POST"])
def classinformation_index():
    form = ClassInformationForm()
    if form.validate_on_submit():
        session['classnumber'] = form.number.data
        return redirect(url_for("classinformation_index"))
    return render_template('classinformation.html', form=form)


################################ MONITOR #########################

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
    
def recognize_from_model(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def add_attendance(student):
    name = student.split('_')[0]
    studentid = student.split('_')[1]
    classnumber = 5
    time = datetime.now().strftime("%H:%M:%S")

    new_attendance=Attendance(name,studentid,classnumber,time)
    db.session.add(new_attendance)
    db.session.commit()

@app.route('/monitor',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('landing.html') 
    cap = cv2.VideoCapture(0)
    ret = True  
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            student = recognize_from_model(face.reshape(1,-1))[0]
            #add_attendance(student)
            cv2.putText(frame,f'Attendance recorded: {student}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()    
    return render_template('monitor.html') 
#### This function will run when we add a new user


############################################# ADD STUDENTS ############################

class AddStudentsForm(FlaskForm):
    name = StringField('Student Name: ',validators=[DataRequired()])
    studentid = IntegerField('Student Id: ',validators=[DataRequired()])
    classnumber = IntegerField('Class Number: ',validators=[DataRequired()])
    time = StringField('Time: ',validators=[DataRequired()])
    submit = SubmitField('Submit')




def datetoday():
    return date.today().strftime("%m_%d_%y")

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

@app.route('/addstudents',methods=['GET','POST'])
def addstudents_index():
    form=AddStudentsForm()
    if form.validate_on_submit():
        session['name'] = form.name.data
        session['studentid'] = form.studentid.data
        session['classnumber'] = form.classnumber.data
        session['time'] = form.time.data
        new_student=Student(session['name'],session['studentid'],session['classnumber'],session['time'])
        image_storage_path = 'static/faces/'+session['name']+'-'+str(session['studentid'])
        if not os.path.isdir(image_storage_path):
            os.makedirs(image_storage_path)
        cap = cv2.VideoCapture(0)
        i,j = 0,0
        while 1:
            _,frame = cap.read()
            faces = extract_faces(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame,f'Images Captured: {i}/20',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = session['name']+'_'+str(i)+'.jpg'
                    cv2.imwrite(image_storage_path+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                j+=1
            if j==200:
                break
            cv2.imshow('Adding new User',frame)
            if cv2.waitKey(1)==27:
                break
        db.session.add(new_student)
        db.session.commit()
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        return redirect(url_for("thankyou_index"))
    return render_template('addstudents.html',form=form) 


########################### View attendance ######################
@app.route('/attendance',methods=['GET','POST'])
def viewattendance_index():

    students=Student.query.all()
    students_attendance=Attendance.query.all()
    print(students_attendance)
    return render_template('viewattendance.html',students=students,students_attendance=students_attendance)

######################## Thank you ##############################
@app.route('/thankyou')
def thankyou_index():
    return render_template('thankyou.html')

@app.route('/tryadd',methods=['GET','POST'])
def tryadd_index():
    form=AddStudentsForm()
    if form.validate_on_submit():
        session['name'] = form.name.data
        session['studentid'] = form.studentid.data
        session['classnumber'] = form.classnumber.data
        session['time'] = form.time.data
        new_student=Student(session['name'],session['studentid'],session['classnumber'],session['time'])
        db.session.add(new_student)
        db.session.commit()
        return redirect(url_for("home"))
    return render_template('tryadd.html',form=form)



################################ DB #############################

class Student(db.Model):
    
    __tablename__ = 'students'

    id=db.Column(db.Integer,primary_key=True)
    studentid=db.Column(db.Integer)
    name=db.Column(db.Text)
    classnumber=db.Column(db.Integer)
    time=db.Column(db.Text)

    def __init__(self,name,studentid,classnumber,time):
        self.name = name
        self.studentid = studentid
        self.classnumber = classnumber
        self.time = time

    def __repr__(self):
        return f"Name: {self.name} Classnumber: {self.classnumber} Time: {self.time} studentid:{self.studentid}"     

class Attendance(db.Model):
    
    __tablename__ = 'attendance'

    id=db.Column(db.Integer,primary_key=True)
    studentid=db.Column(db.Integer)
    name=db.Column(db.Text)
    classnumber=db.Column(db.Integer)
    time=db.Column(db.Text)

    def __init__(self,name,studentid,classnumber,time):
        self.name = name
        self.studentid = studentid
        self.classnumber = classnumber
        self.time = time

    def __repr__(self):
        return f"Name: {self.name} Classnumber: {self.classnumber} Time: {self.time} studentid:{self.studentid}"     




with app.app_context():
    db.create_all()




if __name__ == '__main__':
    app.run(debug=True)