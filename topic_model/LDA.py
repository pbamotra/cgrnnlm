import os
import Corpus
import string
import logging
import numpy as np
from pprint import pprint
from gensim import corpora
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel
from nltk.corpus import BracketParseCorpusReader

# change logging level to DEBUG to see logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logger = logging.getLogger('')
lmtzr = WordNetLemmatizer()
stops = stopwords.words('english')
punct = list(string.punctuation)
extra_waste = set(open('english.stop.txt', 'r').read().split('\n'))
extra_waste.update(["...", "mr.", "ms.", "''", "n't", "year", "market", "company", "stock", "stocks", "million",
                    "billion", "thousand", "price", "prices", "share", "shares", "group", "u.s.", "month",
                    "quarter", "dollar", "dollars", "day", "daily", "week", "weekly", "time", "corp.", "inc.", "profit",
                    "loss", "sale", "make", "made", "business", "up", "down", "rise", "fall", "rose", "fallen", "wa",
                    "ha", "cent", "cents"])
extra_waste.update(range(10))
DOC_LEN_THRESHOLD = 8  # Min. number of sentences in a document
N_TOPICS = 10  # No. of topics for LDA
DICTIONARY_FILE = 'dictionary.dict'
LDA_MODEL_FILE = 'lda.gensim'


def has_no_bad_start(word):
    return not word.startswith('*') \
           and not word.startswith('-') \
           and not word.startswith('`') \
           and not word.startswith("'") \
           and not word.startswith("0")


def clean_text(doc_as_sentences):
    doc_as_sentences = [lmtzr.lemmatize(word.lower()) for word in doc_as_sentences]
    cleaned_text = [word for word in doc_as_sentences
                    if word not in stops
                    and word not in punct
                    and word not in extra_waste
                    and has_no_bad_start(word)]
    return cleaned_text


def train(refresh=True):
    if refresh:
        ptb = BracketParseCorpusReader(Corpus.DATA_DIR, Corpus.FILE_PATTERN)
        train_folders = [str(i) + str(j) for i in range(2) for j in range(10)]
        train_folders += [str(i) + str(j) for i in range(2, 3) for j in range(5)]

        dictionary = corpora.dictionary.Dictionary()
        train_documents = list()

        logger.debug('Starting to parse training documents')
        for folder in train_folders:
            for ptb_file in os.listdir(os.path.join(Corpus.DATA_DIR, folder)):
                document_sentences = ptb.sents(fileids=[os.path.join(folder, ptb_file)])
                if len(document_sentences) > DOC_LEN_THRESHOLD:
                    doc2sentence = list(chain.from_iterable(document_sentences))
                    doc2sentence = clean_text(doc2sentence)
                    dictionary.add_documents([doc2sentence])
                    train_documents.append(doc2sentence)
        logger.debug('Parsed all training documents')

        dictionary.filter_extremes(no_below=1, no_above=0.5)
        dictionary.save(DICTIONARY_FILE)

        logger.debug('Creating corpus for training data')
        corpus = [dictionary.doc2bow(text) for text in train_documents]
        logger.debug('Finished creating corpus')

        logger.debug('Training LDA model on corpus')
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=N_TOPICS, passes=20)
        logger.debug('Completed LDA training')

        lda.save(LDA_MODEL_FILE)
    else:
        dictionary = corpora.dictionary.Dictionary.load(DICTIONARY_FILE)
        lda = LdaModel.load(LDA_MODEL_FILE)

    return lda, dictionary


def get_beta_vector(model, dictionary, word):
    cleaned_word = clean_text([word])
    cleaned_word = cleaned_word if len(cleaned_word) > 0 else None

    log_beta = np.array([0] * N_TOPICS)

    if cleaned_word is not None:
        word_id_cnt = dictionary.doc2bow(cleaned_word)
        if len(word_id_cnt) == 1:
            word_col = word_id_cnt[0][0]
            log_beta = model.expElogbeta[:, word_col]

    return log_beta


if __name__ == '__main__':
    # Set refresh = True to train LDA again
    lda_model, model_dictionary = train(refresh=False)
    pprint(lda_model.show_topics(num_topics=N_TOPICS, num_words=20))
    print
    print 'finance ->', get_beta_vector(lda_model, model_dictionary, 'finance')
    print
    print 'hocus-pocus ->', get_beta_vector(lda_model, model_dictionary, 'hocus-pocus')