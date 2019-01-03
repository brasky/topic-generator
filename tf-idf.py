import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
import os
from preprocessor import preprocess, get_ngrams

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
                    text = preprocess(text_as_string)
                    texts.append(text)
        return texts

    texts = get_texts()
    texts = get_ngrams(texts)

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    index = SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

# Not 100% how to interpret these results
    for i in range(0, len(corpus)):
        print(index.get_similarities(corpus[i]))


    #
    #
    #
    #
    #
    #
    #
