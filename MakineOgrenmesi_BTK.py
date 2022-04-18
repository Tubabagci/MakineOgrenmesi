# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 19:56:32 2022

@author: Asus
"""

#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------KODLAR---------------

#VERİ YÜKLEME
veriler=pd.read_csv('eksikveriler.csv')
print(veriler)
 
#veri on isleme

boy=veriler[["boy"]]
#boy=veriler[['boy']] şeklinde de yazılabilir.
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)

x=10

#SINIF OLUŞTURMA
class insan:
    boy=180
 #FONKSİYON TANIMLAMA
    def kosmak(self,b):
        return b+10

ali=insan()
print(ali.boy)
print(ali.kosmak(90))
    
#LİSTELER
l=[1,2,3]


#EKSİK VERİLER
 
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy='mean') #'nan' olan değerleri ortalama değer ile değiştir(mean->ort için.)

Yas = veriler.iloc[:,1:4].values #Veri kümesinden boy,kilo ve yas klonları çekilir.
print(Yas)

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4]) #yas'ın ort değerini 'nan' değerlere atıyoruz.
print(Yas)


#KATEGORİK VERİLER

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))

sonuc = pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=('boy','kilo','yas'))
print(sonuc2)


cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)


#VERİ KÜMESİNİ TRAİN VR TESST OLARAK BÖLME

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


#ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

