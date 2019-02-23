import pandas as pd
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Other
import re
import string
import numpy as np
import pandas as pd
import math


data=pd.ExcelFile('class2-9.xlsx')

df101=pd.read_excel(data,'101')
df104=pd.read_excel(data,'104')
df501=pd.read_excel(data,'501')
df701=pd.read_excel(data,'701')
df801=pd.read_excel(data,'801')

df101=df101['DESC']
df104=df104['Material Short Description']
df501=df501['Material Short Description']
df701=df701['Material Short Description']
df801=df801['Material Description']

#creating the initial dataframe to be used
df=pd.DataFrame(columns=['DESC','CLASS'])
c=0
d=0
for item in df101:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'101']
        c+=1
    else:
        d+=1
for item in df104:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'104']
        c+=1
    else:
        d+=1
for item in df501:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'501']
        c+=1
    else:
        d+=1
for item in df701:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'701']
        c+=1
    else:
        d+=1
for item in df801:
    if(item!="NOT TO BE USED"):
        df.loc[c]=[item,'801']
        c+=1
    else:
        d+=1
    
#cleaning text

df = df.dropna() # remove missing values
df=df.drop_duplicates()
df = df[df.DESC.apply(lambda x: x !="")]
token_count=0
def clean_text(text):
    global token_count
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
    token_count+=len(text)
    #stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    lemmatized_words = [wordnet.lemmatize(word, pos='v') for word in text]
    text = " ".join(lemmatized_words)

    return text

df['DESC'] = df['DESC'].map(lambda x: clean_text(x))
avg_words_per_sentence=math.ceil(token_count/len(df))

#removing any duplicates left

df=(df.reset_index()).drop('index',axis=1)
n=len(df)
remove_indexes=[]
for x in range(n):
    for y in range(x+1,n):
        if df['DESC'].loc[x]==df['DESC'].loc[y] and x!=y:
            remove_indexes.append(x)
            
            print('yes--  ',x,'-',df['DESC'].iloc[x],'  ----  ',y,'-',df['DESC'].iloc[y])
df=df.drop(remove_indexes)
df=(df.reset_index()).drop('index',axis=1)

#converting to tagged document for Doc2Vec

tagged_data=[TaggedDocument(words=word_tokenize(sentence.lower()),tags=[sentence.lower()]) for sentence in df['DESC']]
df['TAGGED_DATA']=tagged_data

#tagged data contains TaggedDocuments with word having the tokens and the tag as the same as the cleaned sentence,so that we can identidy it later

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
                
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha


#to get a sentence vector,use the cleaned form of the sentence as an index into model.docvecs
#to get vector for df['DESC'][0] use model.docvecs[df['DESC'][0]] or model[df['DESC'][0]]
