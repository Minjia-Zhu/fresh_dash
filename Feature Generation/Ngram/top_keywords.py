
import re
import csv
import sys
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
csv.field_size_limit(100000000)
csv.field_size_limit()
from collections import Counter
import ast
import collections

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
    
#NLTK packages
import nltk
from nltk.tokenize import sent_tokenize
# Define functions for stopwords, bigrams and lemmatization
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['&#160;'])


#### Tokenize Words and Clean-up Text ####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#### Remove Stopwords, Make Bigrams and Lemmatize #####

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def preprocess_reviews(data):
    data = data.review_contents.values.tolist() #review content in a list
    data_words = list(sent_to_words(data)) # remove punctuations    
    # Build the bigram
    bigram = gensim.models.Phrases(data_words, min_count=2, threshold=100) # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_words_nostops = remove_stopwords(data_words) # Remove Stop Words
    data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)# Form Bigrams
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_lemmatized

########## To create bigrams #########
def ngrams_counts(text_list):
    '''
    takes a text and n gram and counts the number of time it occuers in the entire text
    '''
    dict_ngram = {}
    for word_list in text_list:
        #print(word_list)
        for i in range(0, len(word_list)-1):
            #print(word_list[i])
            grtuple= (word_list[i], word_list[i+1])
            if grtuple in dict_ngram:
                dict_ngram[grtuple] = dict_ngram[grtuple] + 1 
            else:
                dict_ngram[grtuple] = 1
    return dict_ngram


def most_common(gram):
    gram = collections.Counter(gram)
    list_words = []
    for k,v in gram.most_common()[:200]:
        list_words.append(k)
    return list_words


######## Merge back to the original dataset ############

def counter(df_reviews):
    '''
    input: df_reviews (dataframe): counting bigram counts in the reviews
    output: df(dataframe): returns a dataframe with ispection_id and bigrams with their counts
    '''
    df = pd.DataFrame()
    for index, row in df_reviews.iterrows():
        reviews = preprocess_reviews(pd.DataFrame(row).T)
        for review in reviews:
            keyword_count = Counter(zip(review, review[1:]))
            keyword_df = pd.DataFrame([keyword_count], columns=keyword_count.keys())
            keyword_df["inspection_id"] = row.inspection_id
        df = df.append(keyword_df)
    return df


def merged_df(df_reviews):
    '''
    Created an external function to reduce unnessarily long dataframes with not required bigrams as columns
    This function considers 50 rows at a time and strips bigram columns that are not present in the keywords
    input: df_reviews (dataframe): dataframe with reviews who's keyword counts we need
    output: df(dataframe): returns a dataframe with inspection_id and keywords with their counts
    '''
    df = pd.DataFrame()
    for i in range(0, df_reviews.shape[0], 50):
        if i+50 > df_reviews.shape[0]:
            new_df = counter(df_reviews[i:df_reviews.shape[0]])
        else:
            new_df = counter(df_reviews[i:i + 50])
        cols = [col for col in new_df.columns if col in keywords] + [('inspection_id', '')]
        new_df = new_df[cols]
        df = df.append(new_df)
    return df



if __name__ == '__main__':

    cutoff = 50

    #Individual review level
    df_instances = pd.read_csv('data/individual_review_inspection_merged_cleaned.csv', sep=None,engine='python')
    #df_instances.rename(index=str, columns={"content": "review_contents"}, inplace=True)
    #df_instances = df_instances[:100]
    good_restaurants = df_instances[(df_instances['violation_code'] != 1) & (df_instances['rating'] >3)]
    bad_restaurants =  df_instances[(df_instances['violation_code'] ==1 & (df_instances['rating'] <4))]
    #good_restaurants_reviews = " ".join([review for review in good_restaurants.review_contents])
    #bad_restaurants_reviews = " ".join([review for review in bad_restaurants.review_contents])
    good_restaurants.to_csv("good_restaurants_pen_v1.csv")
    bad_restaurants.to_csv("bad_restaurants_pen_v1.csv")

    good_restaurants_reviews = preprocess_reviews(good_restaurants)
    bad_restaurants_reviews = preprocess_reviews(bad_restaurants)

    with open("bad_restaurants_reviews.txt", "w") as file:
        file.write(str(bad_restaurants_reviews))

    with open("good_restaurants_reviews.txt", "w") as file:
        file.write(str(good_restaurants_reviews))

    #Good Restaurants
    bigram_good = ngrams_counts(good_restaurants_reviews)
    common_good_bi = most_common(bigram_good)

    #Bad Restaurant
    bigram_bad = ngrams_counts(bad_restaurants_reviews)
    common_bad_bi = most_common(bigram_bad)

    ## Save them
    with open("common_bad_bi.txt", "w") as file:
        file.write(str(common_bad_bi))

    with open("common_good_bi.txt", "w") as file:
        file.write(str(common_good_bi))
    '''
    f = open('common_bad_bi.txt','r')
    common_bad_bi = f.read()
    common_bad_bi = ast.literal_eval(common_bad_bi)
    f.close()

    f = open('common_good_bi.txt','r')
    common_good_bi = f.read()
    common_good_bi = ast.literal_eval(common_good_bi)
    f.close()
    '''
    df_merged = pd.read_csv('data/aggregated_review_inspection_merged_cleaned.csv', sep=None,engine='python')
    keywords = common_bad_bi + common_good_bi

    merged_withkeywords = merged_df(df_merged)
    merged_withkeywords.rename(index=str, columns={('inspection_id', ''): "inspection_id"}, inplace=True)
    merged_withkeywords.to_csv("merged_withkeywords.csv")

    #final dataframe with keywords as columns
    pd.merge(merged_withkeywords.fillna(0), df_merged, on=["inspection_id"]).to_csv("final_df.csv")

