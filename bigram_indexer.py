#!/usr/bin/python3
# pylint: disable=C0111,I0011,E1102

## @file
#
# Script to index n-grams into the ShareSci database.
#

"""
    Create term-document matrix with tf-idf values
"""
import string
import gc
import os
import re
import sys
import tarfile
import psycopg2
import psycopg2.extras
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from scipy.sparse import linalg as LA
from optparse import OptionParser
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)

DOC_TYPES_KEY = {
                    'title': 2,
                    'abstract': 3,
                    'authors': 4
                }

CONN = psycopg2.connect("dbname='sharesci' user='sharesci' host='localhost' password='sharesci'")

## Insert data into the database.
#
# @param sql String
# Format: A SQL INSERT statement as a string
#
# @param data list
# Format: A list of data tuples for the INSERT statement (should be compatible
# 	with Psycopg2's `cursor.execute(...)`
#
# @return None
#
def insert(sql, data):
    cursor = CONN.cursor()
    try:
        psycopg2.extras.execute_values(cursor, sql, data, page_size=10000)
    except psycopg2.Error as error:
        print("Database error occured while executing '", sql, "'", 'Data: ')
        print(len(data), data[:10], data[-10:])
        print(error.diag.message_primary)
    CONN.commit()
    cursor.close()

## Get the size of the database on disk, in bytes
#
# @return (int) the size of the database on disk, in bytes
#
def get_database_size():
    cur = CONN.cursor()
    size = sys.maxsize
    try:
        cur.execute("select pg_database_size('sharesci')")
        size = cur.fetchall()[0][0]
    except psycopg2.Error as error:
        print('Failed to get database size', file=sys.stderr)
        print(error.diag.message_primary)
    CONN.commit()
    cur.close()
    return size

## Get the document IDs in the database for the given text IDs
#
# @param text_ids (1D array-like)
# Format: A list of text IDs (as strings)
#
# @return (dict) a dict with keys being text IDs (as strings) and values being
# 	the corresponding document IDs (as int). Any text IDs for which no
# 	matching document IDs could be found are excluded (i.e., there are no
# 	keys with the value None)
# 
def get_doc_ids(text_ids):
    doc_id_dict = {}
    cursor = CONN.cursor()
    sql = """SELECT text_ids.text_id, d._id
             FROM (VALUES %s) text_ids(text_id)
             INNER JOIN document d
                 ON text_ids.text_id = d.text_id
             GROUP BY text_ids.text_id, d._id"""
    try:
        psycopg2.extras.execute_values(cursor, sql, [(text_id,) for text_id in text_ids],
                                       page_size=10000)
        doc_id_dict = dict(cursor.fetchall())
    except psycopg2.Error as error:
        print('Failed to get doc_id', file=sys.stderr)
        print(error.diag.message_primary)
    CONN.commit()
    cursor.close()
    return doc_id_dict

## Populate the document table from the new documents in the corpus
#
# @param text_ids (list)
# <br>	Format: `[text_id1 (str), text_id2 (str), ...]`
# <br>	-- A list of textual document IDs
#
# @param doc_lengths (list)
# <br>	Format: `[doc1_length (int), doc2_length (int), ...]`
# <br>	-- A list of document lengths. These should correspond to the
# 	`text_ids`, so `doc_lengths[5]` is the length of the TF vector for the
# 	document represented by `text_ids[5]`, etc.
#
def populate_document_table(text_ids, doc_lengths, options):
    excluded_docs = set([])

    if options.get_parent_docs:
        parent_table = np.array([text_id.split('_')[-1] for text_id in text_ids])
        parent_ids_dict = get_doc_ids(parent_table)
        parent_id_keys = parent_ids_dict.keys()
        excluded_docs |= set([i for i in range(len(text_ids)) if parent_table[i] not in parent_id_keys])
        doc_table = [(
            text_ids[i],
            doc_lengths[i][0],
            parent_ids_dict[parent_table[i]],
            DOC_TYPES_KEY[text_ids[i].partition('_')[0]]
            ) for i in range(len(text_ids)) if i not in excluded_docs]
    else:
        doc_table = [(text_ids[i],
                      doc_lengths[i][0],
                      None,
                      1
                     ) for i in range(len(text_ids)) if i not in excluded_docs]

    sql = """INSERT INTO document (text_id, length, parent_doc, type)
            VALUES %s
            ON CONFLICT (text_id) DO UPDATE 
                SET length=EXCLUDED.length"""

    print("Inserting {} records into document table.".format(len(doc_table)))
    insert(sql, doc_table)
    print("Data inserted.")

    return excluded_docs

def populate_tables(raw_tf, text_ids, terms, options):
    """! Populate the idf, document, tf tables.

     Args:
        raw_tf (scipy.sparse): term-document matrix of term counts.
        doc_ids (obj:`list` of :obj:`str`): List of document ids.
        terms (obj:`list` of :obj:`str`): List of terms.

    Returns:
        None
    """
    tfidftransformer = TfidfTransformer(sublinear_tf=True, use_idf=False, norm=None)
    lnc = tfidftransformer.fit_transform(raw_tf)
    print("Calculating document lengths")
    doc_lengths = LA.norm(lnc, axis=1).reshape(-1, 1)
    print("Finished.")

    excluded_docs = set([])

    if options.new_docs:
        excluded_docs |= populate_document_table(text_ids, doc_lengths, options)

    gram_ids = []
    tf_values = []
    bigram_terms = [[term.partition(' ')[0], term.partition(' ')[2]] for term in terms]
    df_values = np.zeros(len(bigram_terms), dtype=np.uint16).tolist()
    bigram_length = len(bigram_terms)

    rows, cols = lnc.nonzero()
    for row, col in zip(rows, cols):
        if row not in excluded_docs:
            df_values[col] += 1 #calculate document frequency

    print("Inserting {} bigrams".format(bigram_length))

    num_inserted = 0
    bigram_batch_size = 100000
    while num_inserted < bigram_length:
        cursor = CONN.cursor()
        try:
            cursor.callproc('insert_bigram_df', [bigram_terms[num_inserted:(num_inserted+bigram_batch_size)], df_values[num_inserted:(num_inserted+bigram_batch_size)]])
            data = cursor.fetchone()
            if data:
                gram_ids += data[0]
            else:
                print("Warning: insert_bigram_df function returned null.")
        except psycopg2.Error as error:
            print(error)
        CONN.commit()
        cursor.close()
        num_inserted += bigram_batch_size
        if bigram_length - num_inserted > 0 and (num_inserted/1000000).is_integer():
            print("{} bigrams remaining.".format(bigram_length - num_inserted))
    print("Data Inserted")
    df_values = None
    bigram_terms = None

    print("Getting doc_ids from text_ids.")
    doc_ids = get_doc_ids(text_ids)
    print("Calculating tf values.")
    for row, col in zip(rows, cols):
        if text_ids[row] in doc_ids.keys():
            tf_values.append([
                gram_ids[col],
                doc_ids[text_ids[row]],
                float(lnc[row, col]/doc_lengths[row])
                ])

    print("Inserting {} rows into tf table".format(len(tf_values)))
    sql = """INSERT INTO tf(gram_id, doc_id, lnc)
             VALUES %s
             ON CONFLICT (gram_id, doc_id) DO UPDATE 
                SET lnc=EXCLUDED.lnc"""
    insert(sql, tf_values)
    print("Data Inserted.")


def load_files(root, mappings):
    """! Load all the regular files from all the archive files

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


def index_terms(token_dict, options):
    print("Calculating raw tf values.")
    #vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w[A-Za-z_-]{1,19}\b', ngram_range=(1, 2))
    #vectorizer = CountVectorizer(tokenizer=lambda x: [token for token in nltk.word_tokenize(x)], ngram_range=(1, 2))
    vectorizer = CountVectorizer(tokenizer=lambda x: [stemmer.stem(token) for token in nltk.word_tokenize(x)], ngram_range=(1, 2))
    raw_tf = vectorizer.fit_transform(token_dict.values())
    print("Calculation of raw tf values complete.")
    terms = vectorizer.get_feature_names()
    doc_ids = np.array(list(token_dict.keys()))

    # Attempt to reduce memory usage by destroying potentially large objects
    vectorizer = None
    gc.collect()

    populate_tables(raw_tf, doc_ids, terms, options)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-d", dest="doc_dir")
    parser.add_option("-m", dest="mapping_file")
    parser.add_option("--new-docs", action="store_true", default=False, dest="new_docs")
    parser.add_option("--get-parent-docs",
                      default=False,
                      action="store_true",
                      dest="get_parent_docs",
                      help="""
For each text doc id, assume the portion following the last '_' is the text doc
id of the parent doc, and update the database accordingly. This only has an
effect if --new-docs is also specified
        """
                     )

    (OPTIONS, ARGS) = parser.parse_args()

    MAX_DATABASE_SIZE = 100*1000*1000*1000  # 100 GB
    if get_database_size() > MAX_DATABASE_SIZE:
        print("Database is too big! Can't fit more data within the limit!", file=sys.stderr)
        sys.exit(1)

    if OPTIONS.doc_dir and OPTIONS.mapping_file:
        mappings = [[], []]
        with open(OPTIONS.mapping_file) as f:
            for m in eval(f.readline()):
                mappings[0].append(m["arXiv_id"])
                mappings[1].append(m["_id"])

        token_dict = load_files(OPTIONS.doc_dir, mappings)
        index_terms(token_dict, OPTIONS)
        print("All done.")
    else:
        print("Please specify path to the folder which contains all .tar.gz")
