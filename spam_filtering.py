# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:14:09 2019

@author: Prateek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('spam.csv' , encoding='latin-1')
dataset= dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset= dataset.rename(columns={"v1":"class", "v2":"text"})

#Cleaning the text
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
N= 5572
corpus=[]

for i in range(0, N):
    message= re.sub('[^a-zA-Z]',' ', dataset['text'][i])
    message= message.lower().split()
    ps= PorterStemmer()
    
    message=[ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    #Converting the list back to string
    message=' '.join(message)
    corpus.append(message)
         

#Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5200)
X= cv.fit_transform(corpus).toarray()

y= dataset.iloc[:, 0].values

#Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)    

#Training the model
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test results
y_pred= classifier.predict(X_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Predicting the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)





