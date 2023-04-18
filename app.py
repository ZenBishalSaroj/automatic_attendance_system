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
import time
import face_recognition



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
if not os.path.isdir('static/faces/unknowns'):
    os.makedirs('static/faces/unknowns')
if not os.path.isdir('static/faces/captured'):
    os.makedirs('static/faces/captured')
if not os.path.isdir('static/faces/images'):
    os.makedirs('static/faces/images')

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
        session['thisclassnumber'] = form.number.data
        return redirect(url_for("classinformation_index"))
    return render_template('classinformation.html', form=form)


################################ MONITOR #########################

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
    
def recognize_from_model(facearray):
   
    if (np.max(result))>0.7:
        print(result)
        return model.predict(facearray)
    else:
        return False

def add_attendance(student):
    name = student.split('-')[0]
    studentid = student.split('-')[1]
    #time = datetime.now(pytz.timezone('America/New_York')).strftime("%H:%M:%S")
    today = date.today()
    #time = datetime.now().strftime("%H:%M:%S")
    for item in Student.query.filter_by(studentid=studentid):
        student_classnumber = item.classnumber
    if int(session['thisclassnumber'])==student_classnumber:
        if Attendance.query.filter_by(studentid=studentid,time=date.today()).count()>0:
            return 'Attendance recorded for today'
        else:
            new_attendance=Attendance(name,studentid,session['thisclassnumber'],today)
            db.session.add(new_attendance)
            db.session.commit()
            return 'Attendance recorded for today'
    else: 
        print('class doesnt match')
        return False
    

@app.route('/monitor',methods=['GET'])
def start():
    if not session.get('classnumber'):
        return redirect(url_for("classinformation_index"))
    known_face_encodings = []
    known_face_names = []
    userlist = os.listdir('static/faces/images')
    for user in userlist:
            img=user.split('-')[0]
            load_img = face_recognition.load_image_file(f'static/faces/images/{user}/{img}_5.jpg')
            face_encodinggt = face_recognition.face_encodings(load_img)[0]
            #known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(f'static/faces/images/{user}/{img}_0.jpg'))[0])
            known_face_encodings.append(face_encodinggt)
            known_face_names.append(user)
    
    cap = cv2.VideoCapture(0)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    ret = True  
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.imwrite('static/faces/captured/'+'cap01.jpg',frame[y:y+h,x:x+w])
            try:
                face = face_recognition.face_encodings(face_recognition.load_image_file('static/faces/captured/'+'cap01.jpg'))[0]
                matches = face_recognition.compare_faces(known_face_encodings,face)
           
            #face = [cv2.resize(frame[y:y+h,x:x+w], (64, 64))]
            #student = recognize_from_model(face.reshape(1,-1))[0]
                name = "Unknown"
                print(matches)
                    
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                else:
                    face_distances = face_recognition.face_distance(known_face_encodings, face)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                student = name
                if student=='Unknown':
                    cv2.imwrite('static/faces/unknowns/'+str(len(next(os.walk('static/faces/unknowns'))[2])+1)+'.jpg',frame[y:y+h,x:x+w])
                else:
                    result = add_attendance(student)
                    if result:
                        cv2.putText(frame,f'Attendance recorded for today. ',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                    else:
                        cv2.putText(frame,'You are in a different classroom.',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            except:
                pass
        process_this_frame = not process_this_frame

        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template('monitor.html') 



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
    # encoding = []
    # names = []
    # userlist = os.listdir('static/faces/images')
    # for user in userlist:
    #         encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(f'static/faces/images/{user}/{user}_0.jpg'))[0])
    #         names.append(user)
    # face_locations = []
    # face_encodings = []
    # face_names = []
    # process_this_frame = True
    return True
    #joblib.dump(knn,'static/face_recognition_model.pkl')

@app.route('/addstudents',methods=['GET','POST'])
def addstudents_index():
    form=AddStudentsForm()
    if form.validate_on_submit():
        session['name'] = form.name.data
        session['studentid'] = form.studentid.data
        session['classnumber'] = form.classnumber.data
        session['time'] = form.time.data
        new_student=Student(session['name'],session['studentid'],session['classnumber'],session['time'])
        image_storage_path = 'static/faces/images/'+session['name']+'-'+str(session['studentid'])
        if not os.path.isdir(image_storage_path):
            os.makedirs(image_storage_path)
        cap = cv2.VideoCapture(0)
        i,j = 0,0
        while 1:
            _,frame = cap.read()
            faces = extract_faces(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame,f'Images Captured: {i}/10',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = session['name']+'_'+str(i)+'.jpg'
                    cv2.imwrite(image_storage_path+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                j+=1
            if j==100:
                break
            cv2.imshow('Adding new User',frame)
            if cv2.waitKey(1)==27:
                break
        db.session.add(new_student)
        db.session.commit()
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        #train_model()
        return redirect(url_for("thankyou_index"))
    return render_template('addstudents.html',form=form) 

############### Train Model ###################
@app.route('/train')
def trainmodel_index():
    if len(next(os.walk('static/faces/images'))[1])<1:
        return render_template('notenough.html')
    train_model()
    return render_template('info.html',msg='Model Training successful. You can now monitor attendance.')

########################### View attendance ######################
@app.route('/attendance',methods=['GET','POST'])
def viewattendance_index():
    if len(next(os.walk('static/faces/images'))[1])<1:
        return render_template('notenough.html')
    students=Student.query.all()
    students_list =[]
    for student in students:
        total_attendance = Attendance.query.filter_by(studentid=student.studentid)
        image_src = f'/faces/images/{student.name}-{student.studentid}/{student.name}_5.jpg'
        students_list.append({"name":student.name,"studentid":student.studentid,"classnumber":student.classnumber,"time":student.time,"attendance_count":total_attendance.count(),"image_src":image_src})
    return render_template('viewattendance.html',students_list=students_list)

######################## Unknowns #########################
@app.route('/unknowns',methods=['GET','POST'])
def unknowns_index():
    if len(next(os.walk('static/faces/unknowns'))[2])==0:
        return redirect(url_for("home"))
    count = len(next(os.walk('static/faces/unknowns'))[2])
    image_list = []
    for i in range(1,count+1):
        image_src = f'/faces/unknowns/{i}.jpg'
        image_list.append({"image_number":i,"image_src":image_src})
    return render_template('unknowns.html',image_list=image_list)

######################## Thank you ##############################
@app.route('/thankyou')
def thankyou_index():
    return render_template('thankyou.html')

######################### Info ##########################
@app.route('/info')
def info_index():
    return render_template('info.html')

@app.route('/tryadd',methods=['GET','POST'])
def tryadd_index():
       # form=AddStudentsForm()
    # if form.validate_on_submit():
    #     session['name'] = form.name.data
    #     session['studentid'] = form.studentid.data
    #     session['classnumber'] = form.classnumber.data
    #     session['time'] = form.time.data
    #     new_student=Attendance(session['name'],session['studentid'],session['classnumber'],session['time'])
    #     db.session.add(new_student)
    #     db.session.commit()
    #     return redirect(url_for("home"))
    # return render_template('tryadd.html',form=form)
    # if Attendance.query.filter_by(studentid=1,time=date.today()).count()>0:
    #     print('record present')
    # else:
    #     print('record not present')
    Attendance.query.filter_by(name='saroj').delete()
    db.session.commit()
    return render_template('info.html',msg='This is a test page.')



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
        return f"{self.name};{self.classnumber};{self.time};{self.studentid}"     

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
