#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:14:42 2020

@author: amirhosein
"""
# Movie Review Analysis

# The purpose of this project is to develope a model to learn attributes of negative and positive reviews and then classify new reviews.
#I use a moview review corpus from nltk library which is labled as negative and positive.

#%% Importing libraries

import nltk, glob, re

#%% Importing data

neg_reviews = glob.glob('movie_reviews/neg/'+'*.txt')

pos_reviews = glob.glob('movie_reviews/pos/'+'*.txt')

len(neg_reviews), len(pos_reviews) #each 1000

neg_reviews[:5]
pos_reviews[:5]


type(neg_reviews) #list

#Now there are two equal lists of positive and negative reviews.
#%% Loading content

neg_texts=[]
pos_texts=[]

for n in neg_reviews:
        o= open(n,encoding='utf-8-sig')
        neg_texts.append(re.sub('[^0-9a-zA-Z]+', ' ', o.read()))

for p in pos_reviews:
        o= open(p,encoding='utf-8-sig')
        pos_texts.append(re.sub('[^0-9a-zA-Z]+', ' ', o.read()))
        
neg_texts[0]
type(neg_texts) #list

#%% Tokenizing

#transforming each review to its words

neg_lower = [n.lower() for n in neg_texts]
pos_lower = [p.lower() for p in pos_texts]

neg_split = [n.split(" ") for n in neg_lower]
pos_split = [p.split(" ") for p in pos_lower]

neg_split[0]
type(neg_split) #list

#%% Filtering

#filtering useless words

from nltk.corpus import stopwords
import string

stops = stopwords.words('english') + list(string.punctuation) + list('0 1 2 3 4 5 6 7 8 9 10')

neg_filtered = [[t for t in n if t not in stops] 
               for n in neg_split]

pos_filtered = [[t for t in p if t not in stops]
              for p in pos_split]

neg_filtered[0]
type(neg_filtered) #list

#%% Stemming

# transforming each words to its stem

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

neg_stems =[[stemmer.stem(t) for t in n] for n in neg_filtered]
pos_stems =[[stemmer.stem(t) for t in p] for p in pos_filtered]

neg_stems
type(neg_stems) #list

#%% Building bag-of-words

# Creating a dictionary of the words for negative and positive reviews seperately.
# This bag-of-words act as features in classifing task.

def build_bow_features(words): return {word:1 for word in words}

neg_features = [(build_bow_features(n),'neg') for n in neg_stems]
pos_features = [(build_bow_features(n),'pos') for n in pos_stems]

neg_features[0]
pos_features[0]

#%% Sentiment Analysis

# Performing sentiment analysis task by applying a naive bayes classifire from nltk.

from nltk.classify import NaiveBayesClassifier

# Consider 80% of data for the tarining
split = 800

# Train the model on negative and positive BOW training set
sentiment_classifier = NaiveBayesClassifier.train(neg_features[:split]+pos_features[:split])

# train error is 94.75% :
nltk.classify.util.accuracy(sentiment_classifier, neg_features[:split]+pos_features[:split])*100

#The accuracy above is mostly a check that nothing went very wrong in the training, the real measure of accuracy is on the remaining 20% of the data that wasn't used in training, the test data:

# Accuracy here is 75% which is pretty good for such a simple model
nltk.classify.util.accuracy(sentiment_classifier, neg_features[split:]+pos_features[split:])*100

#%% the words that mostly identify a positive or a negative review:
    
sentiment_classifier.show_most_informative_features()

#[out]
Most Informative Features
                 offbeat = 1                 pos : neg    =     11.7 : 1.0
                outstand = 1                 pos : neg    =     10.3 : 1.0
                    plod = 1                 neg : pos    =     10.3 : 1.0
                  predat = 1                 neg : pos    =     10.3 : 1.0
               strongest = 1                 pos : neg    =      9.7 : 1.0
                  turkey = 1                 neg : pos    =      9.0 : 1.0
                 incoher = 1                 neg : pos    =      9.0 : 1.0
                    kudo = 1                 pos : neg    =      9.0 : 1.0
                  annual = 1                 pos : neg    =      8.3 : 1.0
                archetyp = 1                 pos : neg    =      8.3 : 1.0
