# Code modified from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
import re
import csv
import sys
import numpy as np
import pandas as pd
from pprint import pprint
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
csv.field_size_limit(sys.maxsize)

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Helper Functions
def pickle_file(object_to_save, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(object_to_save, fp)

def load_pickle_file(filepath) :
    with open(filepath, "rb") as fp: 
        rv = pickle.load(fp)
    return rv

# Preprocessing Helper Functions

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def get_stopwords(customized_stopwords):
		if customized_stopwords:
				with open('yelp_stopwords.txt', 'r') as f:
						stop_words = [line.strip() for line in f]
		else:
				from nltk.corpus import stopwords
				stop_words = stopwords.words('english')
				stop_words.extend(['&#160;'])

		return stop_words

def remove_stopwords(texts, customized_stopwords):
		stop_words = get_stopwords(customized_stopwords)	
		return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
		return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
		"""https://spacy.io/api/annotation"""
		texts_out = []
		for sent in texts:
				doc = nlp(" ".join(sent)) 
				texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
		return texts_out  

# Preprocessing
def preprocessing(df, content_col_name, model_name, customized_stopwords, save_pickle):
		data = df[content_col_name]
		data = data.values.tolist()
		# tokenize
		data_words = list(sent_to_words(data))

		# create bigrams
		bigram = gensim.models.Phrases(data_words, min_count=2, threshold=100)
		bigram_mod = gensim.models.phrases.Phraser(bigram)

		# Remove Stop Words
		data_words_nostops = remove_stopwords(data_words, customized_stopwords)

		# Form Bigrams
		data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops] 

		# Do lemmatization keeping only noun, adj, vb, adv
		data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

		if save_pickle:
				pickle_file(data_lemmatized, 'lda_pickle_files/data_lemmatized_' + model_name)

		print('Preprocessing done.')

		return data_lemmatized

# Build required dictionaries
def get_prerequisites(data_lemmatized, filter_extremes = True):
		id2word = corpora.Dictionary(data_lemmatized)
		
		if filter_extremes:
				id2word.filter_extremes(no_below=10, no_above=0.5)
				id2word.compactify()
		
		# Create Corpus
		texts = data_lemmatized

		# Term Document Frequency
		corpus = [id2word.doc2bow(text) for text in texts]

		print('Get prerequisites done.')

		return id2word, corpus

# Build LDA Model
def lda_modeling(id2word, corpus, model_name, save_pickle = True):
		lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
		if save_pickle:
				pickle_file(lda_model, 'lda_pickle_files/lda_' + model_name)

		print('LDA modeling done.')

		return lda_model

# Print LDA info
def print_key_words(lda_model, num_topics=20, num_words=10):
		print('Topics and Representative Keywords')
		pprint(lda_model.print_topics(num_topics, num_words))

# Other LDA related functions
def find_optimal_number_of_topics(dictionary, corpus, texts, limit=40, start=2, step=4, print = True):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=100)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    if print:
    		x = range(start, limit, step)
    		plt.plot(x, coherence_values)
    		plt.xlabel("Num Topics")
    		plt.ylabel("Coherence score")
    		plt.legend(("coherence_values"), loc='best')
    		plt.show()

#    		for m, cv in zip(x, coherence_values):
#    				print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    return model_list, coherence_values


def main(df, content_col_name, model_name, customized_stopwords, filter_extremes, save_pickle, print_result, find_optimal):
		data_lemmatized = preprocessing(df, content_col_name, model_name, customized_stopwords, save_pickle)
		id2word, corpus = get_prerequisites(data_lemmatized, filter_extremes)
		lda_model = lda_modeling(id2word, corpus, model_name, save_pickle)
		if print_result:
				print_key_words(lda_model)
		if find_optimal:
				find_optimal_number_of_topics(id2word, corpus, data_lemmatized, limit=40, start=2, step=4, print = True)
		return lda_model, corpus, id2word
