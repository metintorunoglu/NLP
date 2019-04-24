#NLP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#Creating Bag of Words Model
#At the begining cv=CountVectorizer() executed, then number of columns at X was 1565
#So decided max_features=1500
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()  
y=dataset['Liked'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#additional score calculations
score=classifier.score(X_test, y_test)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)