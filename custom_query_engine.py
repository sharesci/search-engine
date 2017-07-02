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
			df = float(t[3])
			idf = np.log(1 + (num_docs / df))
			result[term] = (gram_id, idf)
	except psycopg2.Error as err:
		print('Failed to get term DFs', file=sys.stderr)
		print(err.diag.message_primary, file=sys.stderr)
	pg_conn.commit()
	cur.close()
	return result

def query_cosine_similarities(query_tfidf_tuples, max_results=20):
	cur = pg_conn.cursor()
	result = None
	try:
		values_str = ','.join(cur.mogrify('(%s, %s)', (tfidf_tuple[0],tfidf_tuple[1])).decode() for tfidf_tuple in query_tfidf_tuples)
		sql = "SELECT text_id, SUM(lnc*term_ltc) AS similarity FROM tf INNER JOIN idf ON idf._id = term_id INNER JOIN (VALUES {}) AS query_matrix(query_term, term_ltc) ON query_term=idf.term INNER JOIN document ON document._id=doc_id GROUP BY document.text_id ORDER BY similarity DESC LIMIT %s".format(values_str)
		sql = """SELECT text_id,
				SUM(lnc*term_ltc) AS similarity
			FROM tf
			INNER JOIN (VALUES {}) AS query_matrix(query_gram_id, term_ltc)
				ON query_gram_id=tf.gram_id
			INNER JOIN gram
				ON (gram.gram_id = tf.gram_id)
			INNER JOIN document
				ON document._id=doc_id
			GROUP BY document.text_id
			ORDER BY similarity DESC LIMIT %s
		""".format(values_str);
		cur.execute(sql, (max_results,))
		result = cur.fetchall()
	except psycopg2.Error as err:
		print('Failed to get cosine similarities', file=sys.stderr)
		print(err.diag.message_primary, file=sys.stderr)
	pg_conn.commit()
	cur.close()
	return result


def make_query_vector(query_string):
	#query_vectorizer = CountVectorizer(tokenizer=lambda text: [stemmer.stem(token) for token in nltk.word_tokenize(text)], stop_words='english', ngram_range=(2,2))
	#query_raw_vector = query_vectorizer.fit_transform([query])
	#query_terms = query_vectorizer.get_feature_names()
	#query_terms = [(term.partition(' ')[0], term.partition(' ')[2]) for term in query_vectorizer.get_feature_names()]
	#query_vec = [(query_terms[i], query_raw_vector[0, i]) for i in range(len(query_terms))]

	query_tokens = [stemmer.stem(token) for token in nltk.word_tokenize(query_string)]
	query_vec = []
	for tok1 in query_tokens:
		query_vec.append(((tok1, ''), 1))
		for tok2 in query_tokens:
			query_vec.append(((tok1, tok2), 1))
	return query_vec;


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

if __name__ == '__main__':

	query = None
	try:
		while query != 'exit':
			query = input('Type your query: ')
			doc_scores = process_query(query, max_results=20)
			print("The top 20 scores are: ", doc_scores)
	except EOFError as err:
		print('exit')


	pg_conn.close()
	mongo_client.close()

