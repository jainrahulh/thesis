# -*- coding: utf-8 -*-
"""
    Machine Learning Model Training and Dumping the Model as a joblib file.
"""

#import requests

#response = requests.get('https://www.politifact.com/factchecks/list/?page=2&category=coronavirus')
#response.text

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('APIData2000-FB.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

df.head()

df.shape

df.info()

df.isnull().sum(axis = 1)

df.Ruling_Slug.unique()

df = df[df['Ruling_Slug']!= 'no-flip']
df = df[df['Ruling_Slug']!= 'full-flop'] 
df = df[df['Ruling_Slug']!= 'half-flip']
df = df[df['Ruling_Slug']!= 'barely-true']
df.Ruling_Slug.unique()

df.loc[df['Ruling_Slug'] == 'half-true', 'Ruling_Slug'] = 'true'
df.loc[df['Ruling_Slug'] == 'mostly-true', 'Ruling_Slug'] = 'true'
df.loc[df['Ruling_Slug'] == 'mostly-false', 'Ruling_Slug'] = 'false'
df.loc[df['Ruling_Slug'] == 'pants-fire', 'Ruling_Slug'] = 'false'
#df.loc[df['Ruling_Slug'] == 'barely-true', 'Ruling_Slug'] = 'false'
df.Ruling_Slug.unique()

labels = df.Ruling_Slug
labels.unique()

df.shape

print(labels.value_counts(), '\n')

"""
    Required in MacOS due to security issues while downloading nltk package.
"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from wordcloud import WordCloud

stopwords = nltk.corpus.stopwords.words('english')
extendStopWords = ['Say', 'Says']
stopwords.extend(extendStopWords)

true_word_tokens = pd.Series(
    df[df['Ruling_Slug'] == 'true'].Statement.tolist()).str.cat(sep=' ')

wordcloud = WordCloud(max_font_size=200, stopwords=stopwords, random_state=None, background_color='white').generate(true_word_tokens)

plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.show()

false_word_tokens = pd.Series(
    df[df['Ruling_Slug'] == 'false'].Statement.tolist()).str.cat(sep=' ')

wordcloud = WordCloud(max_font_size=200, stopwords=stopwords, random_state=None, background_color='black').generate(false_word_tokens)

plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.show()

x_train,x_test,y_train,y_test=train_test_split(df['Statement'].values.astype('str'), labels, test_size=0.3, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words=stopwords, max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

pa_classifier=PassiveAggressiveClassifier(C=0.5,max_iter=150)
pa_classifier.fit(tfidf_train,y_train)

y_pred=pa_classifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test,y_pred, labels=['true','false'])

from sklearn.metrics import classification_report
print(f"PA Classification Report : \n\n{classification_report(y_test, y_pred)}")

from sklearn.metrics import classification_report
scoreMatrix = []
confusionMatrix = []
classificationMatrix = []

j = [0.20,0.30,0.40]
ratio = ["80:20","70:30","60:40"]
for i in range(3):
  x_train,x_test,y_train,y_test=train_test_split(df['Statement'].values.astype('str'), labels, test_size=j[i], random_state=7)
  tfidf_vectorizer=TfidfVectorizer(stop_words=stopwords, max_df=0.7)
  tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
  tfidf_test=tfidf_vectorizer.transform(x_test)
  pa_classifier=PassiveAggressiveClassifier(C=0.5,max_iter=150)
  pa_classifier.fit(tfidf_train,y_train)
  y_pred=pa_classifier.predict(tfidf_test)
  scoreMatrix.append(accuracy_score(y_test,y_pred))

  print(f'Split Ratio: {ratio[i]}')
  #print(f'Accuracy: {round(scoreMatrix[i]*100,2)}%')
  confusionMatrix.append(confusion_matrix(y_test,y_pred, labels=['true','false']))
  
  print(f"Classification Report: \n{classification_report(y_test, y_pred)}\n\n") 
  classificationMatrix.append(classification_report(y_test, y_pred))

scoreMatrix

confusionMatrix

classificationMatrix

y_test.unique()

labels = ['true', 'false']
cm = confusion_matrix(y_test,y_pred, labels=labels)
print(cm)
import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['true', 'false']); ax.yaxis.set_ticklabels(['false', 'true']);

from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(stop_words=stopwords, max_df=0.7)),
                            ('model', PassiveAggressiveClassifier())])

pipeline.fit(x_train, y_train)

pipeline.predict(x_train)

text = ["higher R in the North West and South West is an important part of moving towards a more localised approach to lockdown"]

pipeline.predict(text)

from joblib import dump

dump(pipeline, filename="news_classifier.joblib")

