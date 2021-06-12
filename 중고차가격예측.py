import sys
import sklearn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# Where to save the figures
PROJECT_ROOT_DIR = "."

###### 데이터 속성명 ['차량번호', '연식', '주행거리', '연료', '변속기', '연비', '차종', '배기량', '색상', '세금미납','압류', '저당', '제시번호', '차량모델', '제조사', '가격']

imputer = SimpleImputer(strategy="median")
OneHot = OneHotEncoder(categories='auto')
encoder = LabelEncoder()


#입력데이터
dataset = pd.read_csv("r_carinfo(중복제거).csv",encoding = 'cp949')


#가격데이터 전처리
dataset['가격']=dataset['가격'].replace('만원','',regex=True)
dataset['가격']=dataset['가격'].replace('[,]','',regex=True)
dataset['가격']=pd.to_numeric(dataset['가격'])

bins=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,np.inf]
labels=[0,1,2,3,4,5,6,7,8,9,10]
dataset['가격'] = pd.cut(dataset['가격'],
                               bins=bins,labels=labels)
dataset['가격']=dataset['가격'].cat.codes

#연식 전처리
dataset['연식']=pd.to_numeric(dataset['연식'].str.split('년',1).str[0])
bins=[i for i in range(0,22)]
labels=[i for i in range(1,22)]
dataset['연식'] = pd.cut(dataset['연식'],
                               bins=bins,labels=labels)
dataset['연식']=dataset['연식'].cat.codes.fillna(0)
dataset['연식'][dataset['연식']==-1]=0

#변속기 전처리
encoder.fit(dataset['변속기'].values)
변속기=encoder.transform(dataset['변속기'].values)
dataset['변속기']=변속기
print("변속기: ",encoder.classes_)
print(dataset['변속기'].unique())
print()

#주행거리전처리
dataset['주행거리']=pd.to_numeric(dataset['주행거리'].replace('[,]','',regex=True))

#배기량전처리
dataset['배기량']=pd.to_numeric(dataset['배기량'].replace('[,]','',regex=True))

#연료 전처리
encoder.fit(dataset['연료'].values)
연료=encoder.transform(dataset['연료'].values)
dataset['연료']=연료
print("연료: ",encoder.classes_)
print(dataset['연료'].unique())
print()

#차종전처리
encoder.fit(dataset['차종'].values)
차종=encoder.transform(dataset['차종'].values)
dataset['차종']=차종
print("차종 ", encoder.classes_)
print(dataset['차종'].unique())
print()


#색상전처리
encoder.fit(dataset['색상'].values)
색상=encoder.transform(dataset['색상'].values)
dataset['색상']=색상
print("색상: ", encoder.classes_)
print(dataset['색상'].unique())
print()

#제조사전처리
encoder.fit(dataset['제조사'].values)
제조사=encoder.transform(dataset['제조사'].values)
dataset['제조사']=제조사
print("제조사: ", encoder.classes_)
print(dataset['제조사'].unique())
print()

#세금미납,압류, 저당 전처리
dataset['저당']=pd.to_numeric(dataset['저당'].str.replace('[가-힣]',''))
dataset['저당'][dataset['저당'].isnull()]=0

dataset['압류']=pd.to_numeric(dataset['압류'].str.replace('[가-힣]',''))
dataset['압류'][dataset['압류'].isnull()]=0

dataset['세금미납']=pd.to_numeric(dataset['세금미납'].str.replace('[가-힣]',''))
dataset['세금미납'][dataset['세금미납'].isnull()]=0

#연비 전처리
dataset['연비']=pd.to_numeric(dataset['연비'].str.replace('[가-힣]',''))
차종별평균연비={}
전체평균연비=0
count=0
for i in dataset['차종'].unique():
    차종별평균연비[i]=round(dataset['연비'][dataset['연비'].notnull()][dataset['차종']==i].mean(),1)
    if math.isnan(차종별평균연비[i])==False:
        전체평균연비=전체평균연비+차종별평균연비[i]
        count=count+1
    if math.isnan(차종별평균연비[i]):
        차종별평균연비[i]=0

전체평균연비 = round(전체평균연비/count,1)
for i in 차종별평균연비:
    if 차종별평균연비[i]==0:
        차종별평균연비[i]=전체평균연비
        
for i in 차종별평균연비.keys():
    dataset['연비'][dataset['차종']==i]=dataset['연비'][dataset['차종']==i].fillna(차종별평균연비[i])


#차량모델 전처리
encoder.fit(dataset['차량모델'].replace(' ','').values)
차량모델=encoder.transform(dataset['차량모델'].values)
dataset['차량모델']=차량모델
print("차량모델: ", encoder.classes_)
print(차량모델)
차량모델라벨={}
for i in encoder.classes_:
    차량모델라벨[i]=encoder.transform([i])

for index, (key, elem) in enumerate(차량모델라벨.items()):
  print(index, key, elem)

train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

y_train = train_set["가격"].copy()
train_set    = train_set.drop("제시번호", axis=1)
train_set    = train_set.drop("가격", axis=1)
X_train    = train_set.drop("차량번호", axis=1)


y_test = test_set["가격"].copy()
test_set    = test_set.drop("제시번호", axis=1)
test_set    = test_set.drop("가격", axis=1)
X_test = test_set.drop("차량번호", axis=1)
    
def insert_data():
    print('차량번호', '연식', '주행거리', '연료', '변속기', '연비', '차종', '배기량', '색상', '세금미납','압류', '저당', '제시번호', '차량모델', '제조사를 입력하세요')
    
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o = input().split()
    new = [(a,b,c,float(d),e,float(f),float(g),h,float(i),j,k,l,m,float(n),float(o))]
    dataset = pd.DataFrame(new, columns = ['차량번호', '연식', '주행거리', '연료', '변속기', '연비', '차종', '배기량', '색상', '세금미납','압류', '저당', '제시번호', '차량모델', '제조사'])

    imputer = SimpleImputer(strategy="median")
    OneHot = OneHotEncoder(categories='auto')
    encoder = LabelEncoder()
    
    #연식 전처리
    dataset['연식']=pd.to_numeric(dataset['연식'].str.split('년',1).str[0])
    bins=[i for i in range(0,22)]
    labels=[i for i in range(1,22)]
    dataset['연식'] = pd.cut(dataset['연식'],
                                   bins=bins,labels=labels)
    dataset['연식']=dataset['연식'].cat.codes.fillna(0)
    dataset['연식'][dataset['연식']==-1]=0

    #변속기 전처리
    encoder.fit(dataset['변속기'].values)
    변속기=encoder.transform(dataset['변속기'].values)
    dataset['변속기']=변속기

    #주행거리전처리
    dataset['주행거리']=pd.to_numeric(dataset['주행거리'].replace('[,]','',regex=True))

    #배기량전처리
    dataset['배기량']=pd.to_numeric(dataset['배기량'].replace('[,]','',regex=True))


    #세금미납,압류, 저당 전처리
    dataset['저당']=pd.to_numeric(dataset['저당'].str.replace('[가-힣]',''))
    dataset['저당'][dataset['저당'].isnull()]=0

    dataset['압류']=pd.to_numeric(dataset['압류'].str.replace('[가-힣]',''))
    dataset['압류'][dataset['압류'].isnull()]=0

    dataset['세금미납']=pd.to_numeric(dataset['세금미납'].str.replace('[가-힣]',''))
    dataset['세금미납'][dataset['세금미납'].isnull()]=0


    dataset = dataset.drop("제시번호", axis=1)
    dataset = dataset.drop("차량번호", axis=1)

    return dataset
    
    
#main program
if __name__ == '__main__':

    #Voting classifiers
    log_clf = LogisticRegression(solver="lbfgs", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(gamma="scale", random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print('Voting:', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)

    #Bagging ensembles
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=200,
        max_samples=100, bootstrap=True, random_state=42)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    print('\nbagging :', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)

    #DecisionTree
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    print('\nDecision Tree:', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)
    
    # Random Forests
    rnd_clf = RandomForestClassifier(n_estimators=200, max_leaf_nodes=16, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred = rnd_clf.predict(X_test)
    print('\nRandom Forests:', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)

    # AdaBoost
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    print('\nAdaBoost:', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)

    # GradientClassifier
    grad_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
    grad_clf.fit(X_train,y_train)
    y_pred = grad_clf.predict(X_test)
    print('\nGradientClassifierBoost:', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)

    # XGBoost
    xgb_clf = XGBClassifier(random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    print('\nXGBoost:', accuracy_score(y_test, y_pred))
    print("y_pred: ",y_pred)
      
    x=insert_data()
    print(x[0:1])
    y_pred = xgb_clf.predict(x[0:1])
    print("y_pred: (1개)",y_pred)
        

    

        
