import os
import sys
import time
import requests as req
from bs4 import BeautifulSoup
from nltk.corpus import BracketParseCorpusReader


FIREFOX_HEADER = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:44.0) Gecko/20100101 Firefox/44.0'
WSJ_URL = 'http://www.cs.cmu.edu/afs/cs/project/cmt-55/lti/Courses/722/Spring-08/Penn-tbank/MRG/WSJ/'
HREF = 'href'
ANCHOR = 'a'
DATA_DIR = 'data'
WSJ = 'WSJ'
FILE_PATTERN = r'.*/WSJ_.*\.MRG'


def make_req(uri):
    _r = req.get(uri, headers={'User-Agent': FIREFOX_HEADER})
    if _r.status_code != 200:
        print "Error Code : " + str(_r.status_code) + "  URL : " + uri
        sys.exit(1)
    return _r


def fetch_wsj_data():
    wsj_page_req = make_req(WSJ_URL)

    wsj_page_soup = BeautifulSoup(wsj_page_req.content, 'lxml')
    section_anchors = wsj_page_soup.find_all(ANCHOR)

    for section_link in section_anchors:
        section_dir = section_link.get(HREF)

        if section_dir[0].isdigit():
            section_dir = os.path.join(DATA_DIR, section_dir[:-1])
            if not os.path.exists(section_dir):
                os.makedirs(section_dir)
            section_href = os.path.join(WSJ_URL, section_dir)

            section_req = make_req(section_href)
            section_soup = BeautifulSoup(section_req.content, 'lxml')
            section_anchors = section_soup.find_all(ANCHOR)

            for document_link in section_anchors:
                document_href = document_link.get(HREF)
                document_link_href = os.path.join(section_href, document_href)

                if document_href.startswith(WSJ):
                    data_req = make_req(document_link_href)
                    with open(os.path.join(section_dir, document_href), 'w') as content_file:
                        content_file.write(data_req.content)
                    time.sleep(1)


def get_sents_by_field_ids(field_ids):
    if not isinstance(field_ids, list):
        field_ids = list(field_ids)
    ptb = BracketParseCorpusReader(DATA_DIR, FILE_PATTERN)
    return ptb.sents(fileids=field_ids)


def print_corpus_metrics(corpus_dir='data'):
    ptb = BracketParseCorpusReader(DATA_DIR, FILE_PATTERN)
    words = ptb.words()
    print 'Total number of words', len(words)
    print 'Total number of unique words', len(set(words))
    print 'Total number of documents', len(ptb.fileids())


if __name__ == '__main__':
    print_corpus_metrics()