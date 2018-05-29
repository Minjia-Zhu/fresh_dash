
import re
import csv
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import ast
import collections
import gensim
from gensim.utils import simple_preprocess
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams


csv.field_size_limit(100000000)
csv.field_size_limit()
    
nlp = spacy.load('en', disable=['parser', 'ner'])
stop_words = stopwords.words('english')
stop_words.extend(['&#160;'])


def tokenize_words(sent_lst,stopwords = True):
    print('Tokenizing words......')
    words_lst = [gensim.utils.simple_preprocess(s, deacc=True) for s in sent_lst] # deacc=True removes punctuations
    if stopwords:
        print('Removing stop words......')
        words_lst = [[w for w in sent if w not in stop_words] for sent in words_lst]
    return words_lst

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
  

def preprocess_reviews(df,topic=False):
    reviews = list(set(df.content.values.tolist()))
    print(len(reviews))
    words_lst_nonstop = tokenize_words(reviews)

    if topic:
        print('Parseing bigram for topic modelling......')
        phrases = gensim.models.phrases.Phrases(remove_stopwords(reviews,stopwords=False),
                                                min_count=2, threshold=100) # higher threshold fewer phrases.
        words_lst_nonstop = [phrases[i] for i in words_lst_nonstop]

    print('Lemmatizing ...... Will take long time .....................')
    lemmatized_text = lemmatization(words_lst_nonstop, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return lemmatized_text

def ngrams_counts(lemmatized_text,n,top_n):
    '''
    takes a list of words, create ngram and counter as dict, also get topn keywords
    '''
    top_words = []

    sents = list(map(lambda x: ' '.join(x), lemmatized_text)) # input is a list of sentences so I map join first
    sents = set(sents)
    ngram = [item for sent in sents for item in ngrams(sent.split(), n)]
    for k,v in Counter(ngram).most_common(top_n):
        top_words.append(k)

    return (ngram,top_words)



if __name__ == '__main__':
    print('Reading individual_review_inspection data......')
    df = pd.read_csv('individual_review_inspection_merged_cleaned.csv')

    d = {'good_hygiene': df[(df.violation_code<1) & (df.rating>3)],
         'bad_hygiene': df[(df.violation_code<1) & (df.rating<3)],
         'good_nonhygiene': df[(df.violation_code==1) & (df.rating>3)],
         'bad_nonhygiene': df[(df.violation_code==1) & (df.rating<3)],
         'hygiene': df[df.violation_code<1],
         'nonhygiene': df[df.violation_code==1]} 
    grams = [1,2,3]

    for n in grams:
        for k, v in d.items():
            lemmatized_text = preprocess_reviews(v,topic=False)
            with open( k+"_lemmatized_"+str(n)+".txt", "w") as file:
                file.write(str(lemmatized_text))

            print('Getting Ngram and most common words......')
            ngram = ngrams_counts(lemmatized_text,n,100)
            with open( k + "_top100_"+str(n)+".txt", "w") as file:
                file.write(str(ngram[1]))
            with open(k + "_" + str(n)+'gram.txt', "w") as file:
                file.write(str(ngram[0]))

