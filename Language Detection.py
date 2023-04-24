# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:07:13 2023

@author: Kunal
"""

# Importing the Libraries

import  pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Importing the Dataset

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
data.head(10)

# Data Cleaning

data.isnull().sum()

data['language'].value_counts()

# Splitting the data for training and testing

text = np.array(data['Text'])
language = np.array(data['language'])

cv = CountVectorizer()
Text = cv.fit_transform(text)

X_train, X_test, y_train, y_test = train_test_split(Text, language, test_size=0.33, random_state=42)

# Training the model

model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Taking an input from the user now

phrase = input("Enter a phrase:")
input = cv.transform([phrase]).toarray()
output = model.predict(input)
print("Language is: ", output)
