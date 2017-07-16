#!/usr/bin/python3

## @file
#
# Script to query the database. It can be used in interactive mode, or
# imported from another script (in which case #perform_query can be used to do
# queries)
#

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from pprint import pprint
import psycopg2
import pymongo
from bson.objectid import ObjectId
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import time

pg_conn = psycopg2.connect("dbname='sharesci' user='sharesci' host='137.148.143.96' password='sharesci'")
mongo_client = pymongo.MongoClient('137.148.143.48', 27017)

mongo_db = mongo_client['sharesci']
papers_collection = mongo_db['papers']

stemmer = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)

## Get the IDF values for the given terms
# 
# @param terms (list-like)
# <br>	Format: A list of terms (each term as str)
# 
# @return (dict) 
# <br>	-- a dict with keys being terms (as str) and values being tuples
# 	of `(gram_id, IDF)`
def get_idfs(terms):
	cur = pg_conn.cursor()
	result = None
	num_docs = 1
	try:
		cur.execute("SELECT COUNT(*) from document")
		num_docs = int(cur.fetchone()[0])
		values_str = ','.join(cur.mogrify('(%s, %s)', (term[0],term[1])).decode() for term in terms)
		sql = """
			SELECT query_term1, query_term2, gram.gram_id, COALESCE(df, 1) 
			FROM (VALUES {}) AS tmp_query(query_term1, query_term2) 
				LEFT OUTER JOIN gram ON (
					term_id_1 = (SELECT term._id FROM term WHERE term.term=query_term1) 
					AND term_id_2 = (SELECT term._id FROM term WHERE term.term=query_term2)
				)
				LEFT OUTER JOIN idf ON (idf.gram_id = gram.gram_id);
			""".format(values_str);
		cur.execute(sql)
		result_tuples = cur.fetchall();
		result = {}
		for t in result_tuples:
			term = (t[0],t[1])
			gram_id = t[2]
			df = float(t[3]) + 1    # Add 1 to protect against div-by-zero
			idf = np.log(1 + (num_docs / df))
			result[term] = (gram_id, idf)
	except psycopg2.Error as err:
		print('Failed to get term DFs', file=sys.stderr)
		print(err.diag.message_primary, file=sys.stderr)
	pg_conn.commit()
	cur.close()
	return result


## Compute the most similar documents, given a query vector
#
# "Similarity" here refers to cosine similarity. This function considers all
# documents, including the sub-documents representing titles, abstracts,
# authors, etc, and computes the cosine similarity of each with the query
# vector. It then takes a weighted sum of grouped documents for the final
# similarity score. A document "group" consists of the main (fulltext)
# document and all sub-documents (i.e., documents which are in a parent-child
# relationship with the main document in the database).
#
# @param query_tfidf_tuples (list)
# <br>	Format: `[(gram_id (int), term_ltc (float)), ...]`
# <br>	-- a query vector containing tuples of gram IDs paired with their 
# 	TF-IDF score
#
# @param max_results (int)
# <br>	-- The maximum number of results to return
#
# @return (list)
# <br>	Format: `[(text_id (str), similarity (float)), ...]`
# <br>	-- A list of tuples containing document IDs paired with similarity 
# 	scores, sorted by similarity.
def query_cosine_similarities(query_tfidf_tuples, max_results=20):
	cur = pg_conn.cursor()
	result = None
	try:
		values_str = ','.join(cur.mogrify('(%s, %s)', (tfidf_tuple[0],tfidf_tuple[1])).decode() for tfidf_tuple in query_tfidf_tuples)
		sql = """
			SELECT (SELECT text_id FROM document d2 WHERE d2._id = dg_id) AS "text_id", similarity 
			FROM (
				SELECT COALESCE(document.parent_doc, document._id) AS "dg_id",
					SUM(lnc*term_ltc*(
						CASE document.type 
							WHEN 1 THEN 0.4 
							ELSE 0.2 
						END)
					) AS similarity
				FROM tf
				INNER JOIN (VALUES {}) AS query_matrix(query_gram_id, term_ltc)
					ON query_gram_id=tf.gram_id
				INNER JOIN gram
					ON (gram.gram_id = tf.gram_id)
				INNER JOIN document
					ON document._id=doc_id
				GROUP BY dg_id
				ORDER BY similarity DESC LIMIT %s
			) AS subquery_1
			;
		""".format(values_str);
		cur.execute(sql, (max_results,))
		result = cur.fetchall()
	except psycopg2.Error as err:
		print('Failed to get cosine similarities', file=sys.stderr)
		print(err.diag.message_primary, file=sys.stderr)
	pg_conn.commit()
	cur.close()
	return result


## Create query vector from a query string
#
# @param query_string (str)
# <br>	-- The query string
#
# @return (list) a vector of query terms. Each term is a tuple of two strings,
# 	representing a bigram (or unigram if the second term is '')
#
def make_query_vector(query_string):
	query_tokens = [stemmer.stem(token) for token in nltk.word_tokenize(query_string)]
	query_vec = []
	for tok1 in query_tokens:
		query_vec.append(((tok1, ''), 1))
		#for tok2 in query_tokens:
		#	query_vec.append(((tok1, tok2), 1))
	return query_vec;


## Process the given query
#
# @param query (str)
# <br>	-- The query string (plain text; just space-separated words)
#
# @param max_results (int) the maximum number of results to return
#
def process_query(query, max_results=20):
	if query is None or not re.match(r'\w', query):
		return

	query_vec = make_query_vector(query)

	term_idfs = get_idfs([v[0] for v in query_vec])

	print("IDF values for terms: ", term_idfs)

	query_tuples = []
	query_l2 = 0.0
	for qterm in query_vec:
		term = qterm[0]
		raw_count = qterm[1]
		tf = 1 + np.log(raw_count) if raw_count != 0 else 0
		tfidf = tf * term_idfs[term][1]
		query_tuples.append([term_idfs[term][0], tfidf])
		query_l2 += tfidf*tfidf
	query_l2 = np.sqrt(query_l2)
	query_tuples = [(tup[0], tup[1]/query_l2) for tup in query_tuples]

	return query_cosine_similarities(query_tuples, max_results=max_results)

## Retrieve metadata for a document
#
# @param results (list)
# <br>	Format: `[(text_id, similarity_score), ...]`
# <br>	-- List of tuples of a document text ID and the corresponding
# 	similarity score (from some query).
#
# @return (list) a list of dicts containing metadata of the document
#
def attach_metadata(results):
	metadata_results = []
	for result in results:
		metadata = {'raw_id': result[0], 'title': '', 'arxiv_id': '', 'score': result[1]}
		if len(result[0]) == 24:
			mongo_result = papers_collection.find({'_id': ObjectId(result[0])})[0]
			metadata['title'] = mongo_result['title'];
			metadata['arxiv_id'] = mongo_result['arXiv_id'];
			metadata['mongo_id'] = result[0];
		else:
			metadata['arxiv_id'] = result[0];
		metadata_results.append(metadata);
	return metadata_results;


## Pretty-print a list of search results
#
# @param results (list)
# <br>	Format: `[(text_id, similarity_score), ...]`
# <br>	-- List of tuples of a document text ID and the corresponding
# 	similarity score (from some query).
#
def pretty_print_metadata_results(results):
	print('{:>2s}  {:100s}  {:15s}  {:7s}        '.format('#', 'Title', 'arXiv id', 'Score'))
	result_num = 1
	for result in results:
		print('{:2d}. {:100s}  {:15s}  {:0.5f}    '.format(result_num, re.sub('[ ]*\n[ ]*', ' ', result['title']), result['arxiv_id'], result['score']))
		result_num += 1


if __name__ == '__main__':

	query = None
	try:
		while query != 'exit':
			times = {}

			query = input('Type your query: ')

			start_time = time.perf_counter()
			doc_scores = process_query(query, max_results=20)
			times['query'] = time.perf_counter() - start_time
			
			start_time = time.perf_counter()
			metadata_results = attach_metadata(doc_scores)
			times['mongo'] = time.perf_counter() - start_time

			print("The top 20 scores are:")
			pretty_print_metadata_results(metadata_results)
			print('\n{:0.4f}s to perform the query, {:0.4f}s to get the metadata for results from Mongo\n'.format(times['query'], times['mongo']))
	except EOFError as err:
		print('exit')


	pg_conn.close()
	mongo_client.close()

