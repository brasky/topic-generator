import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser

def preprocess(text):
    text = remove_stopwords(text)
    return text

def remove_stopwords(text):
    nlp = spacy.load('en')

    my_stop_words = ['say', '’s', '\n', '\n\n', '\'s', 'mr', 'mrs', 'ms', 'ask', 'use', '€', '$']
    text = nlp(text.lower())
    lemma_list = []
    for w in text:
        if  not w.is_stop and not w.is_punct and not w.like_num:
            lemma_list.append(w.lemma_)

    lemma_list = [lemma for lemma in lemma_list if lemma not in my_stop_words]

    return lemma_list

def get_ngrams(texts):
    sentence_stream = [text for text in texts]
    phrases = Phrases(sentence_stream, min_count=1, threshold=1, delimiter=b'_')
    bigram_phraser = Phraser(phrases)
    tokens_ = [bigram_phraser[sent] for sent in phrases[sentence_stream]]
    return tokens_
