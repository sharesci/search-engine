#!/usr/bin/python3
# pylint: disable=I0011,E1102,E1133
"""
    Create term-document matrix with tf-idf values
"""
import string
import os
import sys
import tarfile
import psycopg2
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from scipy.sparse import linalg as LA

CONN = psycopg2.connect("dbname='sharesci' user='sharesci' host='localhost' password='sharesci'")

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return stems

def insert(conn, sql, data):
    cursor = conn.cursor()
    try:
        cursor.executemany(sql, data)
    except psycopg2.Error as error:
        print("Database error occured while executing '", sql, "'")
        print(error.diag.message_primary)
    conn.commit()
    cursor.close()

def populate_document_table(raw_tf, doc_ids):
    """Populate the document table.

     Args:
        raw_tf (scipy.sparse): term-document matrix of term counts.
        doc_ids (obj:`list` of :obj:`str`): List of document ids.

    Returns:
        None
    """
    tfidftransformer = TfidfTransformer(sublinear_tf=True, use_idf=False, norm=None)
    log_tf = tfidftransformer.fit_transform(raw_tf)
    doc_lengths = LA.norm(log_tf, axis=1).reshape(-1, 1)
    doc_table = np.hstack((doc_ids.reshape(-1, 1), doc_lengths))
    sql = """INSERT INTO document VALUES(%s, %s) ON CONFLICT (_id) DO UPDATE"""

    sql = """INSERT INTO document
             VALUES(%s, %s) 
             ON CONFLICT (_id) DO UPDATE 
                SET length=EXCLUDED.length"""

    insert(CONN, sql, doc_table.tolist())

def populate_tf_table(raw_tf, doc_ids, terms):
    """Populate the tf table.

     Args:
        raw_tf (scipy.sparse): term-document matrix of term counts.
        doc_ids (obj:`list` of :obj:`str`): List of document ids.
        terms (obj:`list` of :obj:`str`): List of terms.

    Returns:
        None
    """
    tfidftransformer = TfidfTransformer(sublinear_tf=True, use_idf=False, norm="l2")
    lnc = tfidftransformer.fit_transform(raw_tf)
    tf_table = []
    rows, cols = lnc.nonzero()
    for row, col in zip(rows, cols):
        tf_table.append([terms[col], doc_ids[row], lnc[row, col]])

    sql = """INSERT INTO tf
             VALUES(%s, %s, %s) 
             ON CONFLICT (term, docId) DO UPDATE 
                SET lnc=EXCLUDED.lnc"""

    insert(CONN, sql, tf_table)

def populate_idf_table(raw_tf, terms):
    """Populate the idf table.

     Args:
        raw_tf (scipy.sparse): term-document matrix of term counts.
        doc_ids (obj:`list` of :obj:`str`): List of document ids.
        terms (obj:`list` of :obj:`str`): List of terms.

    Returns:
        None
    """
    tfidftransformer = TfidfTransformer(sublinear_tf=True, use_idf=True, norm="l2")
    tfidftransformer.fit_transform(raw_tf)
    idf_table = []
    i = 0
    while i < len(terms):
        idf_table.append([terms[i], tfidftransformer.idf_[i]])
        i += 1

    sql = """INSERT INTO idf
             VALUES(%s, %s) 
             ON CONFLICT (term) DO UPDATE 
                SET idf=EXCLUDED.idf"""
    insert(CONN, sql, idf_table)

def load_files(root):
    """Load all the regular files from all the archive files

     Args:
        root (obj: `str`): Full path of the folder which contains all .tar.gz files

    Returns:
        None
    """
    token_dict = {}

    for subdir, _, tar_files in os.walk(root):
        for tar_file in tar_files:
            tar_file_path = subdir + os.path.sep + tar_file
            tar = tarfile.open(tar_file_path)
            for member in tar.getmembers():
                file = tar.extractfile(member)
                if file is not None: #only read regular files
                    arxiv_id = tar_file[:4] + "-" + member.name.split("/")[-1][:-4]
                    text = file.read().decode("utf-8")
                    text = text.translate(str.maketrans('', '', string.punctuation))
                    token_dict[arxiv_id] = text

    return token_dict

if __name__ == "__main__":
    if sys.argv[1]:
        TOKEN_DICT = load_files(sys.argv[1])

        VECTORIZER = CountVectorizer(tokenizer=tokenize, stop_words='english')
        RAW_TF = VECTORIZER.fit_transform(TOKEN_DICT.values())
        TERMS = VECTORIZER.get_feature_names()
        DOC_IDS = np.array(list(TOKEN_DICT.keys()))

        populate_document_table(RAW_TF, DOC_IDS)

        populate_tf_table(RAW_TF, DOC_IDS, TERMS)

        populate_idf_table(RAW_TF, TERMS)
    else:
        print("Please specify path to the folder which contains all .tar.gz")
