#!/usr/bin/python3
# pylint: disable=I0011,E1102,E1133
"""
    Create term-document matrix with tf-idf values
"""
import string
import gc
import os
import re
import sys
import tarfile
import psycopg2, psycopg2.extras
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from scipy.sparse import linalg as LA

CONN = psycopg2.connect("dbname='sharesci' user='sharesci' host='localhost' password='sharesci'")

stemmer = PorterStemmer()
def tokenize(text):
    return [stemmer.stem(item) for item in word_tokenize(text)]


def insert(conn, sql, data):
    cursor = conn.cursor()
    try:
        psycopg2.extras.execute_values(cursor, sql, data, page_size=1000)
    except psycopg2.Error as error:
        print("Database error occured while executing '", sql, "'", 'Data: ')
        print(len(data), data[:10], data[-10:])
        print(error.diag.message_primary)
    conn.commit()
    cursor.close()

def get_database_size(conn):
    cur = conn.cursor()
    size = sys.maxsize
    try:
        cur.execute("select pg_database_size('sharesci')")
        size = cur.fetchall()[0][0]
    except psycopg2.Error as error:
        print('Failed to get database size', file=sys.stderr)
        print(error.diag.message_primary)
    conn.commit()
    cur.close()
    return size

def populate_tables(raw_tf, doc_ids, terms):
    """Populate the idf, document, tf tables.

     Args:
        raw_tf (scipy.sparse): term-document matrix of term counts.
        doc_ids (obj:`list` of :obj:`str`): List of document ids.
        terms (obj:`list` of :obj:`str`): List of terms.

    Returns:
        None
    """
    print("Calculating normalized tf values.")
    tfidftransformer = TfidfTransformer(sublinear_tf=True, use_idf=False, norm=None)
    lnc = tfidftransformer.fit_transform(raw_tf)
    print("Calculating document lengths.")
    doc_lengths = LA.norm(lnc, axis=1).reshape(-1, 1)
    doc_table = np.hstack((doc_ids.reshape(-1, 1), doc_lengths))
    print("Calculations of document lengths completed.")
    tf_table = []
    df_table = [[term, 0] for term in terms]
    rows, cols = lnc.nonzero()
    for row, col in zip(rows, cols):
        df_table[col][1] += 1 #calculate document frequency
        tf_table.append([terms[col], doc_ids[row], float(lnc[row, col]/doc_lengths[row])]) #normalized term frequency
    print("Calculation of normalized tf values completed.")

    sql = """INSERT INTO idf
             VALUES %s 
             ON CONFLICT (term) DO UPDATE 
                SET df=idf.df+EXCLUDED.df"""
    print("Inserting data into idf table.")
    insert(CONN, sql, df_table)
    print("Data inserted.")

    sql = """INSERT INTO document
             VALUES %s
             ON CONFLICT (_id) DO UPDATE 
                SET length=EXCLUDED.length"""

    print("Inserting data into document table.")
    insert(CONN, sql, doc_table.tolist())
    print("Data inserted.")

    sql = """INSERT INTO tf
             VALUES %s
             ON CONFLICT (term, docId) DO UPDATE 
                SET lnc=EXCLUDED.lnc"""
    print("Inserting data into tf table.")
    insert(CONN, sql, tf_table)
    print("Data inserted")


def load_files(root, mappings):
    """Load all the regular files from all the archive files

     Args:
        root (obj: `str`): Full path of the folder which contains all .tar.gz files

    Returns:
        None
    """
    token_dict = {}

    for subdir, _, tar_files in os.walk(root):
        print("Processing Files\n")
        for tar_file in tar_files:
            if tar_file.endswith(".tar.gz"):
                tar_file_path = subdir + os.path.sep + tar_file
                tar = tarfile.open(tar_file_path)
                for member in tar.getmembers():
                    file = tar.extractfile(member)
                    if file is not None: #only read regular files
                        doc_id = re.sub(r'.preproc$', '', os.path.basename(member.name))
                        if doc_id in mappings[0]:
                            doc_id = mappings[1][mappings[0].index(doc_id)]
                        print("Processing {0}".format(member.name), end="\r")
                        text = file.read().decode("utf-8")
                        text = text.translate(str.maketrans('', '', string.punctuation))
                        token_dict[doc_id] = text
        print("Processing Complete.")

    return token_dict

if __name__ == "__main__":
    MAX_DATABASE_SIZE = 140*1000*1000*1000  # 140 GB
    if get_database_size(CONN) > MAX_DATABASE_SIZE:
        print("Database is too big! Can't fit more data within the limit!", file=sys.stderr)
        sys.exit(1)
    if sys.argv[1]:
        mappings = [[], []]
        with open('./results2.json') as f:
            data = eval(f.readline())
            for d in data:
                mappings[0].append(d["arXiv_id"])
                mappings[1].append(d["_id"])

        TOKEN_DICT = load_files(sys.argv[1], mappings)
        print("Calculating raw tf values.")
        VECTORIZER = CountVectorizer(token_pattern=r'(?u)\b\w[A-Za-z_-]{1,19}\b', stop_words='english')
        RAW_TF = VECTORIZER.fit_transform(TOKEN_DICT.values())
        print("Calculation of raw tf values complete.")
        TERMS = VECTORIZER.get_feature_names()
        DOC_IDS = np.array(list(TOKEN_DICT.keys()))

        # Attempt to reduce memory usage by destroying potentially large objects
        TOKEN_DICT = None
        VECTORIZER = None
        mappings = None
        gc.collect()

        populate_tables(RAW_TF, DOC_IDS, TERMS)

        print("All done.")
    else:
        print("Please specify path to the folder which contains all .tar.gz")
