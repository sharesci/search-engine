#!/usr/bin/python3

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from pprint import pprint
import psycopg2
import pymongo
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys

pg_conn = psycopg2.connect("dbname='sharesci' user='sharesci' host='137.148.143.96' password='sharesci'")
mongo_client = pymongo.MongoClient('localhost', 27017)

mongo_db = mongo_client['sharesci']
papers_collection = mongo_db['papers']

stemmer = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)

def get_idfs(terms):
	cur = pg_conn.cursor()
	result = None
	num_docs = 1
	try:
		cur.execute("SELECT COUNT(*) from document")
		num_docs = int(cur.fetchone()[0])
		values_str = ','.join(cur.mogrify('(%s)', (term,)).decode() for term in terms)
		cur.execute("SELECT query_term, COALESCE(df, 1) FROM idf RIGHT OUTER JOIN (VALUES {}) as tmp_query(query_term) ON term=query_term".format(values_str))
		result_tuples = cur.fetchall();
		result = {}
		for t in result_tuples:
			term = t[0]
			df = float(t[1])
			idf = np.log(1 + (num_docs / df))
			result[term] = idf
	except psycopg2.Error as err:
		print('Failed to get term DFs', file=sys.stderr)
		print(err.diag.message_primary, file=sys.stderr)
	pg_conn.commit()
	cur.close()
	return result

def query_cosine_similarities(query_tfidf_tuples, doc_ids, max_results=20):
	cur = pg_conn.cursor()
	result = None
	try:
		values_str = ','.join(cur.mogrify('(%s, %s)', (tfidf_tuple[0],tfidf_tuple[1])).decode() for tfidf_tuple in query_tfidf_tuples)
		sql = "SELECT text_id, SUM(lnc*term_ltc) AS similarity FROM tf INNER JOIN idf ON idf._id = term_id INNER JOIN (VALUES {}) AS query_matrix(query_term, term_ltc) ON query_term=idf.term INNER JOIN document ON document._id=doc_id WHERE document.text_id IN %s GROUP BY document.text_id ORDER BY similarity DESC LIMIT %s".format(values_str)
		cur.execute(sql, (tuple(doc_ids), max_results))
		result = cur.fetchall()
	except psycopg2.Error as err:
		print('Failed to get term DFs', file=sys.stderr)
		print(err.diag.message_primary, file=sys.stderr)
	pg_conn.commit()
	cur.close()
	return result

def process_query(query):
	if query is None or not re.match(r'\w', query):
		return

	mongo_result = papers_collection.find({'$and':[{'$text': {'$search': query}}]}, {'_id': 1, 'title': 1, 'authors': 1, 'updated-date': 1, 'score': {'$meta': 'textScore'}})

	query_vectorizer = CountVectorizer(tokenizer=lambda text: [stemmer.stem(token) for token in nltk.word_tokenize(text)], stop_words='english')
	query_raw_vector = query_vectorizer.fit_transform([query])
	query_terms = query_vectorizer.get_feature_names()

	term_idfs = get_idfs(query_terms)

	print("IDF values for terms: ", term_idfs)

	query_tuples = []
	query_l2 = 0.0
	rows, cols = query_raw_vector.nonzero()
	for col in sorted(cols):
		term = query_terms[col]
		raw_count = query_raw_vector[0, col]
		tf = 1 + np.log(raw_count) if raw_count != 0 else 0
		tfidf = tf * term_idfs[term]
		query_tuples.append([term, tfidf])
		query_l2 += tfidf*tfidf
	query_l2 = np.sqrt(query_l2)
	query_tuples = [(tup[0], tup[1]/query_l2) for tup in query_tuples]
	document_ids = [str(item['_id']) for item in mongo_result]

	doc_scores = query_cosine_similarities(query_tuples, document_ids)
	print("The top 20  scores are: ", doc_scores)

if __name__ == '__main__':

	query = None
	try:
		while query != 'exit':
			query = input('Type your query: ')
			process_query(query)
	except EOFError as err:
		print('exit')


	pg_conn.close()
	mongo_client.close()

