# https://shirishkadam.com/tag/spacy/
# https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/

import spacy
from collections import Counter
nlp = spacy.load('en')


# Predefined variable
# all_tags = {w.pos: w.pos_ for w in doc}
noisy_pos_tags = ['PROP','DT','IN']
min_token_length = 2

def generate_features(document, nlp):
    doc = nlp(document)
    sent_list = list(doc.sents)
    sent = sent_list[0]

    bigram = []
    root_token = ""
    pos = ""
    nbor_pos = ""
    word = ""

    for token in sent:
        if not isNoise(token):
            pos = token.tag_
            word = token.text
            bigram.append(token.text)
            bigram.append(str(doc[token.i + 1]))
            nbor_pos = doc[token.i + 1].tag_
        if token.dep_ == "ROOT":
            root_token = token.tag_
    return p

# Helper function
def isNoise(token):
    '''
    Check if the token is a noise or not
    '''
    is_noise = False
    if token.pos_ in noisy_pos_tags: # check for pre-defineed noisy tag
        is_noise = True
    elif token.is_stop == True: # check for stop words
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    return is_noise

    

def cleanup(token, lower = True):
    if lower:
       token = token.lower()
    return token.strip()

def pos_words (sentence, token, ptag):
    '''
    check all tags used with a word
    '''
    # extract all review sentences that contains the term - token
    sentences = [sent for sent in sentence.sents if token in sent.string.lower()]
    pwrds = []
    for sent in sentences:
        for word in sent:
            if token in word.string:
                pwrds.extend([child.string.strip() for child in word.children
                if child.pos_ == ptag])
    return Counter(pwrds).most_common(10)


# top unigrams used in the reviews
cleaned_list = [cleanup(word.string) for word in doc if not isNoise(word)]
Counter(cleaned_list).most_common(5)

# Check all adj used with the term food
pos_words(doc, 'food', 'ADJ')
