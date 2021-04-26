#Offensive Language indentification in tweets
## An automated model trained to detect whether a tweet is offensive or not on multiple levels of classification.

Digital bullying is happening on a daily basis and all of us are facing in form or another on social media. 

Proposing a solution to tackle this problem by generating an automated tool that uses ML algorithmns and techniques to detect these offensive languages in our cases specifically in tweets and then decide whether the tweet was targeted, and if so, classifying the target into more labels 

# Tools or Technoligies used 
- Scikit Learn
- NLTK
- TQDM

## Classifiers trained or experimented on 

- DecisionTree Algorithm
- SVC
- RandomForest
- KNN
- Multinomial Naive Bayes

## Results
SVC model has outperformed all the other models and the results are as follows for each level of classification.

	Predicting whether the tweet is Offensive or not:
		Training Accuracy : 0.98284
		Test Accuracy : 0.856

	Predicting whether the tweet is Targeted or UnTargeted:
		Training Accuracy : 0.66412
		Test Accuracy : 0.69406

	Classifing the Targeted Insult tweet to Individual (IND), Group (GRP), Others (OTH):
		Training Accuracy : 0.98636
		Test Accuracy : 0.69387


## Installation and Execution

Your python envirmonment should be installed with proper libraries such as Scikit_learn, NLTK, TQDM, MatPlotLib. You can pip or brew commands depending on the machine you are using to download these libaries

	git clone https://github.com/prajwal2495/Offensive_tweet_identification.git
	cd Offensive_tweet_identification
	python Offensive_tweets_detection.py

Note:
Every train on the levels of classification results in a confusion matrix being displayed on the screen, close the confusion matrix to continue with the execution.
