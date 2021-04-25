#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

import pprint
pp = pprint.PrettyPrinter(indent=5)


# In[2]:


print("reading data set....")
training_data_set = pd.read_csv("/Users/prajwalkrishn/Desktop/My_Computer/project - Dsci 601/Offensive_Tweet_Detection/Dataset/MOLID.csv")
print("Done reading....")


# In[3]:


training_data_set.head(5)


# In[4]:


tweets = training_data_set[["tweet"]]
level_A_labels = training_data_set[["subtask_a"]]
level_B_labels = training_data_set.query("subtask_a == 'Offensive'")[["subtask_b"]]
level_C_labels = training_data_set.query("subtask_b == 'TIN'")[["subtask_c"]]

All_Cleaned_tweets = copy.deepcopy(tweets)


# In[5]:


##Data Cleaning and Pre-Processing


# In[6]:


tweets.head(5)


# In[7]:


level_A_labels.head(5)


# In[8]:


level_B_labels.head(5)


# In[9]:


level_C_labels.head(5)


# In[10]:


All_Cleaned_tweets.head(5)


# In[11]:


import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer,WordNetLemmatizer
lancaster = LancasterStemmer()
wordNet = WordNetLemmatizer()


# In[14]:


def remove_webTags_UserNames_Noise(tweet):
    things_to_be_removed_from_tweets = ['URL','@USER','\'ve','n\'t','\'s','\'m']
    
    for things in things_to_be_removed_from_tweets:
        tweet = tweet.replace(things,'')
    
    return re.sub(r'[^a-zA-Z]', ' ', tweet)

def stop_words_removal(tokens):
    cleaned_tokens = []
    stop = set(stopwords.words('english'))
    for token in tokens:
        if token not in stop:
            if token.replace(' ','') != '':
                if len(token) > 1:
                    cleaned_tokens.append(token)
    return cleaned_tokens

def tokenize(tweet):
    lower_cased_tweet = tweet.lower()
    return word_tokenize(lower_cased_tweet)

def stemming(tokens):
    cleaned_tokens = []
    for token in tokens:
        token = lancaster.stem(token)
        if len(token) > 1:
            cleaned_tokens.append(token)
    return cleaned_tokens

def lemmatization(tokens):
    cleaned_tokens = []
    for token in tokens:
        token = wordNet.lemmatize(token)
        if len(token) > 1:
            cleaned_tokens.append(token)
    return cleaned_tokens


# In[15]:


tqdm.pandas(desc = "clean...")
All_Cleaned_tweets['tweet'] = tweets['tweet'].progress_apply(remove_webTags_UserNames_Noise)

tqdm.pandas(desc="Tokenize..")
All_Cleaned_tweets['tokens'] = All_Cleaned_tweets['tweet'].progress_apply(tokenize)

tqdm.pandas(desc="remove STOPWORDS...")
All_Cleaned_tweets['tokens'] = All_Cleaned_tweets['tokens'].progress_apply(stop_words_removal)

tqdm.pandas(desc="Stemming...")
All_Cleaned_tweets['tokens'] = All_Cleaned_tweets['tokens'].progress_apply(stemming)

tqdm.pandas(desc="Lemmatize...")
All_Cleaned_tweets['tokens'] = All_Cleaned_tweets['tokens'].progress_apply(lemmatization)

text_vector = All_Cleaned_tweets['tokens'].tolist()


# In[16]:


All_Cleaned_tweets.head(5)


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer

def tfid(text_vector):
    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors
  
def get_vectors(vectors, labels, keyword):
    if len(vectors) != len(labels):
        print("Unmatching sizes!")
        return
    result = list()
    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
    return result


# In[18]:


vectors_level_a = tfid(text_vector) # Numerical Vectors A
labels_level_a = level_A_labels['subtask_a'].values.tolist() # Subtask A Labels

vectors_level_b = get_vectors(vectors_level_a, labels_level_a, "Offensive") # Numerical Vectors B
labels_level_b = level_B_labels['subtask_b'].values.tolist() # Subtask B Labels

vectors_level_c = get_vectors(vectors_level_b, labels_level_b, "TIN") # Numerical Vectors C
labels_level_c = level_C_labels['subtask_c'].values.tolist() # Subtask C Labels


# In[19]:


pp.pprint(vectors_level_a)


# In[20]:


pp.pprint(labels_level_c)


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_a[:], labels_level_a[:], train_size=0.70)

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_b[:], labels_level_b[:], train_size=0.75)

train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

print("fit begins...")
warnings.filterwarnings(action='ignore')
classifier = DecisionTreeClassifier(max_depth=800, min_samples_split=5)
params = {'criterion':['gini','entropy']}
classifier = GridSearchCV(classifier, params, cv=3, n_jobs=4)
classifier.fit(train_vectors, train_labels)
classifier = classifier.best_estimator_
print("fit complete....")

print("calculating accuracy....")
accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
print("Training Accuracy:", accuracy)
test_predictions = classifier.predict(test_vectors)
accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:", )
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels,test_predictions))


# In[ ]:


print("SVM model experiment begins ...")
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_a[:], labels_level_a[:], train_size=0.70)

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_b[:], labels_level_b[:], train_size=0.75)

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

classNames = np.unique(test_labels)
print("fit begins...")
warnings.filterwarnings(action='ignore')
classifiersvc = SVC()
print(classifiersvc.get_params().keys())
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
classifierGrid = GridSearchCV(classifiersvc, param_grid, refit = True, verbose=2)
classifierGrid.fit(train_vectors, train_labels)
classifierGrid = classifierGrid.best_estimator_
print("fit complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels, classifierGrid.predict(train_vectors))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors)
accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:", )
matrix = confusion_matrix(test_labels, test_predictions)
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels,test_predictions))

plottedCM = plot_confusion_matrix(classifierGrid, test_vectors, test_labels,display_labels=classNames, cmap=plt.cm.Blues)
plt.show()


# In[ ]:


print("RandomForest model experiment begins ...")
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_a[:], labels_level_a[:], train_size=0.70)

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_b[:], labels_level_b[:], train_size=0.75)

train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

print("fit begins...")
warnings.filterwarnings(action='ignore')
classifierRFC = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
print(classifierRFC.get_params().keys())
param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}
classifierGrid = GridSearchCV(classifierRFC, param_grid, refit = True, verbose=2)
classifierGrid.fit(train_vectors, train_labels)
classifierGrid = classifierGrid.best_estimator_
print("fit complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels, classifierGrid.predict(train_vectors))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors)
accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:", )
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels,test_predictions))


# In[ ]:


print("MNB model experiment begins ...")
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_a[:], labels_level_a[:], train_size=0.70)

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_b[:], labels_level_b[:], train_size=0.75)

train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

print("fit begins...")
warnings.filterwarnings(action='ignore')
classifierMNB = MultinomialNB()
# print(classifierMNB.get_params().keys())
param_grid = { 
    'alpha': [1, 10, 100, 1000],
    'fit_prior': [True, False]
}
classifierGrid = GridSearchCV(classifierMNB, param_grid, refit = True, verbose=2, n_jobs=2)
classifierGrid.fit(train_vectors, train_labels)
classifierGrid = classifierGrid.best_estimator_
print("fit complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels, classifierGrid.predict(train_vectors))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors)
accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:", )
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels,test_predictions))


# In[ ]:


print("KNN model experiment begins ...")
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_a[:], labels_level_a[:], train_size=0.70)

#train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_b[:], labels_level_b[:], train_size=0.75)

train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

print("fit begins...")
warnings.filterwarnings(action='ignore')
classifierKNN = KNeighborsClassifier()
#print(classifierKNN.get_params().keys())
param_grid = { 
    'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'weights': ['uniform', 'distance'],
    'n_jobs': [-1]
}
classifierGrid = GridSearchCV(classifierKNN, param_grid, refit = True, verbose=2, n_jobs=2)
classifierGrid.fit(train_vectors, train_labels)
classifierGrid = classifierGrid.best_estimator_
print("fit complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels, classifierGrid.predict(train_vectors))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors)
accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:", )
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels,test_predictions))


# In[ ]:




