# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("satislar.csv")
#test
print(veriler)

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

'''
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
 
sc=StandardScaler()

X_train = sc.fit_transform(x_train) #değerler normalize ediliyor
X_test = sc.transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)
'''
# Model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train) # aylara(x_train)göre satış değerlerini(y_train)i tahmin edicek.

tahmin=lr.predict(x_test) #x_teste karşılık satış değerlerini tahmin edicek.

x_train = x_train.sort_index() #index'e göre sıralar.
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))#x_test^teki değerlerin linear regression'daki karşılıklarını gösterir.
plt.title("Aylara Göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")