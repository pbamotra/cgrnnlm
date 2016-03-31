import os
import Corpus
import string
import logging
from gensim import corpora
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel
from nltk.corpus import BracketParseCorpusReader

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('')
lmtzr = WordNetLemmatizer()
stops = stopwords.words('english')
punct = list(string.punctuation)
extra_waste = set(open('english.stop.txt', 'r').read().split('\n'))
extra_waste.update(["...", "mr.", "ms.", "''", "n't", "year", "market", "company", "stock", "stocks", "million",
                    "billion", "thousand", "price", "prices", "share", "shares", "group", "u.s.", "month",
                    "quarter", "dollar", "dollars", "day", "daily", "week", "weekly", "time", "corp.", "inc.", "profit",
                    "loss", "sale", "make", "made", "business", "up", "down", "rise", "fall", "rose", "fallen", "wa",
                    "ha"])
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

# Set refresh = True to train LDA again
def train(refresh=False):
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

        dictionary.filter_extremes(no_below=5, no_above=0.8)
        dictionary.save(DICTIONARY_FILE)

        n_words = len(dictionary.token2id)

        logger.debug('Creating corpus for training data')
        corpus = [dictionary.doc2bow(text) for text in train_documents]
        logger.debug('Finished creating corpus')

        logger.debug('Training LDA model on corpus')
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=N_TOPICS, passes=20)
        logger.debug('Completed LDA training')

        lda.save(LDA_MODEL_FILE)
    else:
        dictionary = corpora.dictionary.Dictionary.load(DICTIONARY_FILE)
        n_words = len(dictionary.token2id)
        lda = LdaModel.load(LDA_MODEL_FILE)

    # TODO: Return normalised vectors from beta matrix
    # lda.show_topics(num_topics=N_TOPICS, num_words=n_words, log=True, formatted=True)

if __name__ == '__main__':
    train()