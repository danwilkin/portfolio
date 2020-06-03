# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
dataset = pd.read_csv('amazon_alexa.tsv', delimiter = '\t', quoting= 3)

#add a length column to increase models accuracy
dataset['length'] = dataset['verified_reviews'].apply(len)

#EDA phase
plt.figure(figsize= (40,15))
plt.subplot(2,2,1)
sns.violinplot(data = dataset, x = 'feedback', y = 'rating')

plt.subplot(2,2,2)
sns.countplot(x = dataset['feedback'])

plt.subplot(2,2,3)
sns.barplot(data = dataset, x = 'variation', y = 'rating', palette = 'deep')

plt.subplot(2,2,4)
sns.countplot(x = dataset['rating'])
plt.show()

dataset.feedback.value_counts()
dataset.variation.value_counts()
dataset.rating.value_counts()

#NLP - creating our bag of words
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 3150):
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2100)
X = cv.fit_transform(corpus).toarray()

# making our X and y variables
X = pd.DataFrame(X)
X = pd.concat([X, dataset['length']],axis =1)
y = dataset.iloc[:, 4].values

# splitting data into train and test sets
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# fitting our model to a random forest algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion= 'entropy')
classifier.fit(X_train, y_train)

# predicting our test set results
from sklearn.metrics import confusion_matrix, classification_report
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
plt.show()

# predicting our train set results
y_pred_train =  classifier.predict(X_train)
cm_train =  confusion_matrix(y_train, y_pred_train)

sns.heatmap(cm_train, annot = True)
plt.show()

# checking our classification report for train/test sets
cr_test = classification_report(y_test, y_pred)
cr_train = classification_report(y_train, y_pred_train)