# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

with open('win.pickle','rb') as file:
    datalist1=pickle.load(file)
    #print(datalist1['ml']['command'][1])

data={}
ball=[]
ball_speedx=[]
ball_speedy=[]
direction=[]
platform=[]
command=[]
#print(datalist1['ml']['scene_info'][1]['ball'])

for i in range(0,len(datalist1['ml']['scene_info'])-1):    
    ball.append(datalist1['ml']['scene_info'][i]['ball'])
    ball_speedx.append(datalist1['ml']['scene_info'][i]['ball'][0]-datalist1['ml']['scene_info'][i+1]['ball'][0])
    ball_speedy.append(datalist1['ml']['scene_info'][i]['ball'][1]-datalist1['ml']['scene_info'][i+1]['ball'][1])
    platform.append(datalist1['ml']['scene_info'][i]['platform'])
    command.append(datalist1['ml']['command'][i])

    if ball_speedx[i] > 0:
        if ball_speedy[i] > 0:
        #lower right
            direction.append(0)
        else:
        #upper right
            direction.append(1)
    else:
        if ball_speedy[i] > 0:
        #lower left
            direction.append(2)
        else:
        #upper left
            direction.append(3)
       

#preprocess platform
PlatX=np.array(platform)[:,0]

#preprocess ball
BallX=np.array(ball)[:,0]
BallY=np.array(ball)[:,1]

Ball_speedX=np.array(ball_speedx)
Ball_speedY=np.array(ball_speedy)

Direction=np.array(direction)

#preprocess commend
Command=np.array(command)

for i in range(0,len(Command)):
    if(Command[i]=='NONE'):
        Command[i]=0
    elif(Command[i]=='MOVE_LEFT'):
        Command[i]=-1
    elif(Command[i]=='MOVE_RIGHT'):
        Command[i]=1
    elif(Command[i]=='SERVE_TO_LEFT'):
        Command[i]=2
    elif(Command[i]=='SERVE_TO_RIGHT'):
        Command[i]=3
        
#transform to csv
import pandas as pd
data=[[],[],[],[],[],[],[]]
for i in range(0,len(BallX)):
    data[0].append(BallX[i])
for i in range(0,len(BallY)):
    data[1].append(BallY[i])
for i in range(0,len(Ball_speedX)):
    data[2].append(Ball_speedX[i])
for i in range(0,len(Ball_speedY)):
    data[3].append(Ball_speedY[i])
for i in range(0,len(PlatX)):
    data[4].append(PlatX[i])
for i in range(0,len(Direction)):
    data[5].append(Direction[i])  
for i in range(0,len(Command)):
    data[6].append(Command[i])     
data=(list(map(list, zip(*data))))
csv=pd.DataFrame(data=data)
csv.to_csv('testcsv.csv',encoding='gbk')

#preprocessing
dataset=csv
feature=dataset.iloc[:,[0,1,2,3,4,5]].values
answer=dataset.iloc[:,6].values
feature=np.asarray(feature,'int32')
answer=np.asarray(answer,'int32')

#資料劃分
x_train, x_test, y_train, y_test = train_test_split(feature, answer, test_size=0.3, random_state=9)

classifier=RandomForestClassifier(random_state=0)
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)


#save model
filename="model.pickle"
pickle.dump(classifier,open(filename, 'wb'))
