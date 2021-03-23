#!/usr/bin/env python
# coding: utf-8

# # Detecting Fake News
# 
# Do you trust all the news you hear from social media?
# All news are not real, right?
# 
# So how will you detect fake news?
# The answer is Python. We can easily make a difference between real and fake news by using advanced Python techniques.
# 
# In order for us to program, we need to be aware of key terms like:
# 
# 1. **Fake news** 
# 2. **Tfidfvectorizer**
# 3. **PassiveAggressive Classifer**

# ## What is Fake News?
# 
# A type of yellow journalism, fake news encapsulate pieces of news that may be hoaxes and is generaly spread through social media and other online media. This is often done to further or impose certain ideas and is often achieved with political agendas. Such news items may contain false and/or exaggerated claims, and may end up being viralized by algorithms, and users may end up in a filter bubble.
# 

# ## What is a TFidfVectorizer?
# 
# **TF(Term Frequency):** The number of times a word appears in a documents is tis Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.
# 
# **IDF(Inverse Document Frequency):** Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus(collection).
# 
# The Tfidfvectorizer converts a collection of raw documents into a matrix of TF-IDF features.

# ## What is a PassiveAggressive Classifier?
# 
# Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing verylittle change in the norm of the weight vector.

# # Objective:
# 
# Our objective is to build a model to aaccurately classify a piece of news as REAL or FAKE. 
# 
# So, how are we going to be able to build the model?
# 
# First, we are going to be using sklearn to build a TFidfVectorizer on our dataset. Then, we initialize a PassiveAgressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

# # The Dataset:
# 
# The dataset we'll use for this project- we'll call it news.csv. This dataset has a shape of 7796 x 4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE.

# ## The Setup
# 
# First we need to install the pandas, numpy, and sklearn libraries with pip:

# In[1]:


pip install numpy pandas sklearn


# Now we import our newly installed libraries

# In[2]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# # The Process:
# 
# #1. Let's read the data into a DataFrame, and get the shape of the data and the first 5 records 

# In[3]:


#Read the data
df = pd.read_csv('C:\\Users\\cordzzy\\Desktop\\fake news project\\news.csv')

#Get shape and head
df.shape
df.head()


# # 

# #2. Get the labels from the DataFrame

# In[4]:


#Get the Labels
labels = df.label
labels.head()


# # 

# #3. Split the dataset into training and testing sets

# In[5]:


#Split the dataset
x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 7)


# # 

# #4. Let's initialize a TFidfVectorizer with stop words from the English language and a maximum document frequency of 0.7(terms with a higher document frequency will be discarded). Stop words are the most common words in a language that are to be filtered out before processing the natural language data. And a TFidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features. 
#     
# Now, fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.

# In[6]:


#Initialize a TFidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

#Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# # 

# #5. Next, we'll initialize a PassiveAggressive Classifier. We'll fit this on tfidf_train and y_train.
# 
# Then we'll predict on the test set from the TFidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics.

# In[7]:


#Initialize a PassiveAggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# # 

# #6. We got an accuracy of 92.5% with this model. Finally, let's print out a confusion matrix to gain insight into the number of false and true negatives and positives.

# In[9]:


#Build confusion matrix
confusion_matrix(y_test,y_pred, labels = ['FAKE','REAL'])


# # 

# So with this model, we have 586 true positives, 586 true negatives, 43 false positives, and 52 false negatives.

# # Summary:
# 
# ***We learned to detect fake news with Python. We took a political dataset, implemented a TFidfVectorizer, initialized a PassiveAgressiveClassifier, and fit our model. We ended up obtaining an accuracy of 92.5% in magnitude.***
