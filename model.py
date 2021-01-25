import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import pandas as pd

df = pd.read_csv("crimes_df7.csv")


df.fillna(df.median(), inplace=True)
df = df.dropna()
df.reset_index()
df = pd.get_dummies(df,columns=['PATROL_BORO'])
df = pd.get_dummies(df,columns=['VIC_SEX'])
df = pd.get_dummies(df,columns=['VIC_RACE'])
import datetime 
import calendar
result_month=[]
result_year=[]
result_day=[]
for i in df['CMPLNT_FR_DT']:
    month,day,year=i.split('/')
    result_month.append(month)
    result_year.append(year)
    result_day.append(day)
df['month']=result_month
df['year']=result_year
df['day']=result_day
VIC_AGE_GROUP_mapping ={'25-44':0, '45-64':1, '<18':2, '18-24':3, 'UNKNOWN':4, '65+':5,'1014':6, '-931':7, '-972':8}
df['VIC_AGE_GROUP']=df['VIC_AGE_GROUP'].map(VIC_AGE_GROUP_mapping)
df = df[df.VIC_AGE_GROUP != 8]
df = df[df.VIC_AGE_GROUP != 7]
df = df[df.VIC_AGE_GROUP != 6]
all_crimes = ["assualt","grand larency","petit larency",
        "harrasment","exposed to weapons","criminal crimes","public safty crimes","administrative crimes","vehical crimes",
        "drugs and alcaholic crimes","theif and robbery ","kidnapping","frauds","children crimes"]
df_mapping ={#assualt
           'ASSAULT 3 & RELATED OFFENSES':0,'FELONY ASSAULT':0,'OFFENSES AGAINST THE PERSON':0,
             #grand larency
             'GRAND LARCENY':1, 
             #petit larency
             'PETIT LARCENY':2,'PETIT LARCENY OF MOTOR VEHICLE':2,
             #harrasment
            'HARRASSMENT 2':3,
            #sex crime
           'SEX CRIMES':4,'RAPE':4,'PROSTITUTION & RELATED OFFENSES':4,
             #exposed to weapons
            'DANGEROUS WEAPONS':5, 'BURGLARY':5,
             #criminal crimes
       'CRIMINAL MISCHIEF & RELATED OF':7,'CRIMINAL TRESPASS':7,
        #public safty crimes
       'OFF. AGNST PUB ORD SENSBLTY &':9, 'OFFENSES AGAINST PUBLIC SAFETY':9, 'LEWDNESS,PUBLIC':9, 'ARSON':9,'JOSTLING':9, 
        #administrative crimes
       'MISCELLANEOUS PENAL LAW':11, 'OFFENSES AGAINST PUBLIC ADMINI':11,  'ADMINISTRATIVE CODE':11,
        #vehical crimes     
       'VEHICLE AND TRAFFIC LAWS':13, 'DISORDERLY CONDUCT':13,'HOMICIDE-NEGLIGENT-VEHICLE':13,'UNAUTHORIZED USE OF A VEHICLE':13,'GRAND LARCENY OF MOTOR VEHICLE':13, 'INTOXICATED & IMPAIRED DRIVING':13, 
        #drugs and alcaholic crimes 
        'DANGEROUS DRUGS':16,'ALCOHOLIC BEVERAGE CONTROL LAW':16,
        #theif and robbery
       'OTHER OFFENSES RELATED TO THEF':17,'THEFT-FRAUD':17,'THEFT OF SERVICES':17, 'ROBBERY':17,'POSSESSION OF STOLEN PROPERTY':17,"BURGLAR'S TOOLS":17,
       #kidnapping
       'KIDNAPPING & RELATED OFFENSES':27,
        #frauds
       'FORGERY':30,'FRAUDULENT ACCOSTING':30, 'FRAUDS':30,'OFFENSES INVOLVING FRAUD':30,
         #childern crimes
        'OFFENSES RELATED TO CHILDREN':37,'CHILD ABANDONMENT/NON SUPPORT':37,
       'AGRICULTURE & MRKTS LAW-UNCLASSIFIED':41,'OTHER STATE LAWS (NON PENAL LA':36,'NYS LAWS-UNCLASSIFIED FELONY':29,'FALSE REPORT UNCLASSIFIED':47,'HOMICIDE-NEGLIGENT,UNCLASSIFIE':52, 'THEFT OF SERVICES, UNCLASSIFIE':54,'LOITERING':43, 'OTHER STATE LAWS':55,'NOISE,UNECESSARY':49, 'TAX LAW':50, 'NYS LAWS-UNCLASSIFIED VIOLATION':51,'ENDAN WELFARE INCOMP':45,'GAMBLING':35}

df['OFNS_DESC']=df['OFNS_DESC'].map(df_mapping)
df = df[df.OFNS_DESC != 43]
df = df[df.OFNS_DESC != 48]
df = df[df.OFNS_DESC != 49]
df = df[df.OFNS_DESC != 50]
df = df[df.OFNS_DESC != 55]
df = df[df.OFNS_DESC != 52]
df = df[df.OFNS_DESC != 54]
df = df[df.OFNS_DESC != 41]
df = df[df.OFNS_DESC != 51]

df = df[df.OFNS_DESC != 36]
df = df[df.OFNS_DESC != 45]
df = df[df.OFNS_DESC != 47]
df = df[df.OFNS_DESC != 29]
df = df[df.OFNS_DESC != 35]
L=["morning","afternoon","evening","night"]
result=[]
for x in df.CMPLNT_FR_TM:
        
    times=x.split(':')
    if int(times[0])>6 and int(times[0])<=12:
        result.append(L[0])
    elif int(times[0])>12 and int(times[0])<=17:
        result.append(L[1])
    elif int(times[0])<20:
        result.append(L[2])
    else : 
        result.append(L[3])
df['time_zone']=result

time_zone_mapping ={'afternoon':0, 'morning':1, 'evening':2, 'night':3}
df['time_zone']=df['time_zone'].map(time_zone_mapping)

df.dropna()
df = df.fillna(df.median())

from sklearn.model_selection import train_test_split

#Split dataset to Training Set & Test Set
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)
Features=['Latitude', 'Longitude',
       'VIC_AGE_GROUP', 'VIC_SEX_D', 'VIC_SEX_E',
       'VIC_SEX_F', 'VIC_SEX_M', 'VIC_RACE_AMERICAN INDIAN/ALASKAN NATIVE',
       'VIC_RACE_ASIAN / PACIFIC ISLANDER', 'VIC_RACE_BLACK',
       'VIC_RACE_BLACK HISPANIC', 'VIC_RACE_UNKNOWN', 'VIC_RACE_WHITE',
       'VIC_RACE_WHITE HISPANIC', 'month', 'year', 'day', 'time_zone']
Target= ['OFNS_DESC']

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test

print('Feature Set Used    : ', Features)
print('Target Class        : ', Target)
print('Training Set Size   : ', x.shape)
print('Test Set Size       : ', y.shape)


# Random Forest
# Create Model with configuration
from sklearn.ensemble import RandomForestClassifier,VotingClassifier

rf_model = RandomForestClassifier(n_estimators=30, # Number of trees
                                  min_samples_split = 20,
                                  bootstrap = True, 
                                  max_depth = 50, 
                                  min_samples_leaf = 25)

# Model Training
rf_model.fit(X=x1,y=x2)

# Prediction
result = rf_model.predict(y[Features])


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
# Model Evaluation
ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("========== Random Forest Results ==========")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)

from flask import Flask , render_template , request , flash , redirect , url_for
from folium.plugins import Draw
import json
import folium
import datetime

def getCoords(coords):  
    for i in range(len(coords)):
        if(coords[i] == '['):
            start = i 
        if(coords[i] == ']'):
            end = i+1
            break 

    coords = coords[start+1 : end-1]
    pair = coords.split(",")
    x = float(pair[0])
    y = float(pair[1])        
    return x , y 
def getAgeGroup(age):
    try : 
        age = int(age)
        if(age>64):
            return 5
        if(age<18):
            return 2
        if(age>19 and age<25):
            return 3
        if(age>24 and age<45):
            return 0
        if(age>44 and age<65):
            return 1

    except : 
        return 4  
def getSex(sexe):
    d = 0 
    m = 0 
    f = 0
    e = 0
    if(sexe=='D'):
        d=1
    if(sexe=='F'):
        f=1
    if(sexe=='M'):
        m=1
    return d,m,f,e
def getRaces(race):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    if(race=='a'):
        a = 1
    if(race=='b'):
        b = 1
    if(race=='c'):
        c = 1
    if(race=='d'):
        d = 1
    if(race=='e'):
        e = 1
    if(race=='f'):
        f = 1
    if(race=='g'):
        g = 1
    return a , b, c , d, e , f , g
def getTime(time):
    time = str(time)
    tmp = time.split(':')[0]
    if int(tmp)>6 and int(tmp)<=12:
        return 1
    elif int(tmp)>12 and int(tmp)<=17 :
        return 0
    elif int(tmp)<20:
        return 2
    else : 
        return 3
def getKeyFromValue(arg):
    t = []
    for k , v in df_mapping.items():
        if v == arg :
            t.append(k)
    return t 
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    coords = ''
    if request.method == "POST":
        coords = request.form.get('coords')
        name = request.form.get('name')
        race = request.form.get('race')
        sexe = request.form.get('sexe')
        age = request.form.get('age')
        trip_time_str = request.form.get('trip_time')
        trip_time = datetime.datetime.strptime(trip_time_str, '%Y-%m-%dT%H:%M')
        Latitude, Longitude = getCoords(coords)
        VIC_AGE_GROUP = getAgeGroup(age)
        VIC_SEX_D, VIC_SEX_M, VIC_SEX_F, VIC_SEX_E = getSex(sexe)
        a , b , c , d , e , f , g  = getRaces(race)
        time = getTime(trip_time.time())
        input = [Latitude, Longitude, VIC_AGE_GROUP, VIC_SEX_D, VIC_SEX_E, VIC_SEX_F,
       VIC_SEX_M, a , b , c , d , e , f , g , trip_time.month, trip_time.year, trip_time.day, time]
        coords = [Latitude , Longitude ]
        input = [input]
        result1 = rf_model.predict(input)
        proba = rf_model.predict_proba(input)
        CrimesList = dict(zip(all_crimes, proba[0]))
        return render_template("result.html", crimes = CrimesList)
        
    return render_template("index.html", coords = coords )

if __name__ == '__main__':
    app.run(debug=True)