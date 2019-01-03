import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import os
from preprocessor import preprocess, get_ngrams
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Get dataset


# Texts are sorted so that every two articles share the same topic.
# ie. the ideal number of topics found should be 3
# and article numbers should be sorted as so: (1, 2), (3, 4), (5, 6)
directory_path = "articles"
directory_files = os.listdir(directory_path)

if __name__ == "__main__":

    def get_texts():
        texts = []
        for file in directory_files:
            file_path = os.path.join(directory_path, file)
            if file_path.endswith(".txt"):
                with open(file_path, encoding='UTF-8') as text:
                    text_as_string = ""
                    for line in text:
                        text_as_string += line
                    text = preprocess(text_as_string) #Gets lemmas, etc.
                    texts.append(text)
        return texts

    texts = get_texts() # Opens all articles in directory and appends to list
    texts = get_ngrams(texts) #Finds collocative phrases

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    max_topics = len(texts) #The maximum number of potential topics.

    corpus_texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]

    def get_model(dictionary, corpus, max_topics): #Here's the beast
        best_lda_model = None
        best_score = None
        best_num_topics = 1
        current_num_topics = 1

        for i in range(1, max_topics):
            lda_model = LdaModel(corpus=corpus, num_topics=current_num_topics, id2word=dictionary, passes=1000)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus_texts, dictionary=dictionary, coherence='c_v')

            coherence_lda = coherence_model_lda.get_coherence()
            current_perplexity_score =  lda_model.log_perplexity(corpus)
            current_score = coherence_lda #New variable because I was playing around with using some function of the coherence and perplexity, but just went with coherence
            print("Topics: ", current_num_topics, "Perplexity Score: ", current_perplexity_score, "Coherence Score: ", coherence_lda)

            # Saves the model with the highest score
            if best_score == None or current_score > best_score:
                best_score = current_score
                best_lda_model = lda_model
                best_num_topics = current_num_topics

            current_num_topics += 1

        print("\nBest Num Topic: ", best_num_topics ,best_score)
        return best_lda_model

    lda_model = get_model(dictionary, corpus, max_topics)

    # Get dataset
    directory_name = "articles"
    directory_listdir = os.listdir(directory_name)

    articles_sorted_by_topic = {}
    for i in range(0, max_topics):
        articles_sorted_by_topic[i] = []

    print("\nArt. - (Topic Num, Probability)")
    for i in range(0, max_topics):
        print(i+1, lda_model[corpus[i]])


    #
    #
    #
    #
    #
    #
    #
    #
