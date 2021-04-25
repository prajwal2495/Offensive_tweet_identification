import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer,WordNetLemmatizer

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

import pprint
pp = pprint.PrettyPrinter(indent=5)





print("reading data set....")
training_data_set = pd.read_csv("/Users/prajwalkrishn/Desktop/My_Computer/project - Dsci 601/Offensive_Tweet_Detection/Dataset/MOLID.csv")
print("Done reading....")


tweets = training_data_set[["tweet"]]
level_A_labels = training_data_set[["subtask_a"]]
level_B_labels = training_data_set.query("subtask_a == 'Offensive'")[["subtask_b"]]
level_C_labels = training_data_set.query("subtask_b == 'TIN'")[["subtask_c"]]

All_Cleaned_tweets = copy.deepcopy(tweets)


lancaster = LancasterStemmer()
wordNet = WordNetLemmatizer()



def remove_webTags_UserNames_Noise(tweet):
	''' 
	Removes the username tags which start with the special character "@" on Twitter.
	# @param tweet String that contains the tweet.
	# @return the tweet containing only alphabets, both lowercase and uppercase.
	'''
    things_to_be_removed_from_tweets = ['URL','@USER','\'ve','n\'t','\'s','\'m']
    
    for things in things_to_be_removed_from_tweets:
        tweet = tweet.replace(things,'')
    
    return re.sub(r'[^a-zA-Z]', ' ', tweet)


def stop_words_removal(tokens):
    '''
    Removes stop words from the tweets.
    @param tokens
    '''
    cleaned_tokens = []
    stop = set(stopwords.words('english'))
    for token in tokens:
        if token not in stop:
            if token.replace(' ','') != '':
                if len(token) > 1:
                    cleaned_tokens.append(token)
    return cleaned_tokens

def tokenize(tweet):
    '''
    Tokenises a tweet into words.
    @param tweet String containing the tweet to be tokenised.
    @return word separated from the tweet.
    '''
    lower_cased_tweet = tweet.lower()
    return word_tokenize(lower_cased_tweet)

def stemming(tokens):
	'''
	Stems a passed list of token to their base word. "Eatable", "Eaten" will become "Eat".
	@param tokens a list of tokens that are to be stemmed.
	@return cleaned_tokens returns a list of stemmed tokens.
	'''
    cleaned_tokens = []
    for token in tokens:
        token = lancaster.stem(token)
        if len(token) > 1:
            cleaned_tokens.append(token)
    return cleaned_tokens

def lemmatization(tokens):
	'''
	Converts a word to it's base word.
	@param tokens a list of tokens that are to be stemmed.
	@return cleaned_tokens returns a list of stemmed tokens.
	'''
    cleaned_tokens = []
    for token in tokens:
        token = wordNet.lemmatize(token)
        if len(token) > 1:
            cleaned_tokens.append(token)
    return cleaned_tokens



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




def tfid(text_vector):
	'''
	Converts the passed raw document to td-idf feature form.
	Uses fit() and transform() on a TdidfVectorizer object.
	@param text_vector tweets to be converted into tf-idf feature form.
	@return vectors returns the tweets converted into tf-idf features.
	'''
    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors
  
def get_vectors(vectors, labels, keyword):
	'''
	Returns a matrix for vectors. Zips vectors and labels IF and only if length of vector list is the same as length of the labels list. 
	Else, the function gets terminated.
	@param vectors These are the vectors for a given label.
	@param labels These are the label values for the given label.
	@param keyword which is the label to annotate for.
	'''
    if len(vectors) != len(labels):
        print("Unmatching sizes!")
        return
    result = list()
    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
    return result



vectors_level_a = tfid(text_vector) # Numerical Vectors A
labels_level_a = level_A_labels['subtask_a'].values.tolist() # Subtask A Labels

vectors_level_b = get_vectors(vectors_level_a, labels_level_a, "Offensive") # Numerical Vectors B
labels_level_b = level_B_labels['subtask_b'].values.tolist() # Subtask B Labels

vectors_level_c = get_vectors(vectors_level_b, labels_level_b, "TIN") # Numerical Vectors C
labels_level_c = level_C_labels['subtask_c'].values.tolist() # Subtask C Labels




print("SVM model experiment begins on Level A classification ...")

train_vectors_level_A, test_vectors_level_A, train_labels_level_A, test_labels_level_A = train_test_split(vectors_level_a[:], labels_level_a[:], train_size=0.70)

classNames = np.unique(test_labels_level_A)
print("Training begins on Level A classification...")
warnings.filterwarnings(action='ignore')
classifiersvc = SVC()
print(classifiersvc.get_params().keys())
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
classifierGrid = GridSearchCV(classifiersvc, param_grid, refit = True, verbose=2)
classifierGrid.fit(train_vectors_level_A, train_labels_level_A)
classifierGrid = classifierGrid.best_estimator_
print("Training complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels_level_A, classifierGrid.predict(train_vectors_level_A))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors_level_A)
accuracy = accuracy_score(test_labels_level_A, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
matrix_level_A = confusion_matrix(test_labels_level_A, test_predictions)
print(matrix_level_A)
print(classification_report(test_labels_level_A,test_predictions))

plottedCM = plot_confusion_matrix(classifierGrid, test_vectors_level_A, test_labels_level_A,display_labels=classNames, cmap=plt.cm.Blues)
plt.show()





train_vectors_level_B, test_vectors_level_B, train_labels_level_B, test_labels_level_B = train_test_split(vectors_level_b[:], labels_level_b[:], train_size=0.75)

classNames = np.unique(test_labels_level_B)
print("Training begins on Level B classification...")
warnings.filterwarnings(action='ignore')
classifiersvc = SVC()
print(classifiersvc.get_params().keys())
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
classifierGrid = GridSearchCV(classifiersvc, param_grid, refit = True, verbose=2)
classifierGrid.fit(train_vectors_level_B, train_labels_level_B)
classifierGrid = classifierGrid.best_estimator_
print("Training complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels_level_B, classifierGrid.predict(train_vectors_level_B))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors_level_B)
accuracy = accuracy_score(test_labels_level_B, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
matrix_level_B = confusion_matrix(test_labels_level_B, test_predictions)
print(matrix_level_B)
print(classification_report(test_labels_level_B,test_predictions))

plottedCM = plot_confusion_matrix(classifierGrid, test_vectors_level_B, test_labels_level_B, display_labels=classNames, cmap=plt.cm.Blues)
plt.show()




train_vectors_level_C, test_vectors_level_C, train_labels_level_C, test_labels_level_C = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

classNames = np.unique(test_labels_level_C)
print("Training begins on Level C classification...")
warnings.filterwarnings(action='ignore')
classifiersvc = SVC()
print(classifiersvc.get_params().keys())
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
classifierGrid = GridSearchCV(classifiersvc, param_grid, refit = True, verbose=2)
classifierGrid.fit(train_vectors_level_C, train_labels_level_C)
classifierGrid = classifierGrid.best_estimator_
print("Training complete....")


print("calculating accuracy....")
accuracy = accuracy_score(train_labels_level_C, classifierGrid.predict(train_vectors_level_C))
print("Training Accuracy:", accuracy)
test_predictions = classifierGrid.predict(test_vectors_level_C)
accuracy = accuracy_score(test_labels_level_C, test_predictions)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
matrix_level_C = confusion_matrix(test_labels_level_C, test_predictions)
print(matrix_level_C)
print(classification_report(test_labels_level_C,test_predictions))

plottedCM = plot_confusion_matrix(classifierGrid, test_vectors_level_C, test_labels_level_C, display_labels=classNames, cmap=plt.cm.Blues)
plt.show()
