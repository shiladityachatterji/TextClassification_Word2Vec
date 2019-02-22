# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Word2Vec
from gensim.models import Word2Vec

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Other
import re
import string
import numpy as np
import pandas as pd


df = pd.read_csv('yelp_labelled.csv', sep = '\t', names = ['text','stars'], error_bad_lines = False)

df = df.dropna() # remove missing values
df = df[df.stars.apply(lambda x: x.isnumeric())]
df = df[df.stars.apply(lambda x: x !="")]
df = df[df.text.apply(lambda x: x !="")]

#print(df.describe())

#print(df.head())

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 2]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)#convert multiple whitespaces into single whitespace
    
    text = text.split()
    #stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    lemmatized_words = [wordnet.lemmatize(word, pos='v') for word in text]
    text = " ".join(lemmatized_words)

    return text

df['text'] = df['text'].map(lambda x: clean_text(x))

df['tokenized'] = df.apply(lambda row : nltk.word_tokenize(row['text']), axis=1) #apply funtion to each row
# building the vocabulary and training the model
model_w2v = Word2Vec(df['tokenized'], size=50, min_count=1, window=4)
model_w2v.train(df['tokenized'], total_examples=len(df['tokenized']), epochs=10)
#print(list(model_w2v.wv.vocab))

vector_dim = 50
'''sentences = df['text'].values
labels = df['stars'].values
'''
#Preparing a word embedding matrix
embedding_matrix = np.zeros((len(model_w2v.wv.vocab), vector_dim))
for i in range(len(model_w2v.wv.vocab)):
    embedding_vector = model_w2v.wv[model_w2v.wv.index2word[i]] #returns the 50-d vector for the word at index2word[i]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#At this point,we have the word vectors,now we need the sentence vectors for classification
#we do this by iterating over every sentence and taking an average of the words for each sentence

XX=[]
yy=[]
for row in range(len(df)):
    sentence_vec_temp=np.zeros((1,50))
    word_count=0
    for word in  df['tokenized'].iloc[row]:
        word_count+=1
        sentence_vec_temp+=model_w2v.wv[word]
    sentence_vec_temp/=word_count
    XX.append(sentence_vec_temp)    
    yy.append(int(df['stars'].iloc[row]))    
    
#Classification
X=np.reshape(np.asarray(XX),(1000,50))
y=(np.asarray(yy))

from sklearn.neural_network import MLPClassifier
MLP=MLPClassifier(hidden_layer_sizes=(300,100,),tol=0.001)
MLP=MLP.fit(X,y)        


    
    

