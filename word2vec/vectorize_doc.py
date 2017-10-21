#!/usr/bin/python3

import numpy as np
import pickle
import os
import re
import json
import sys
import scipy.sparse
from NumpyEmbeddingStorage import NumpyEmbeddingStorage
from QueryEngineCore import QueryEngineCore
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--vectorizer_type', dest='vectorizer_type', action='store', type=str, default='word2vec')
parser.add_argument('--no-tsv-output', dest='tsv_output', action='store_false',  default=True)
cmdargs = parser.parse_args(sys.argv[1:])

datasource = 'cranfield'

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');

with open(os.path.join(data_dir, 'token2id.pickle'), 'rb') as f:
	token2id = pickle.load(f);

with open(os.path.join(data_dir, 'id2freq.npy'), 'rb') as f:
	id2freq = np.load(f);

with open(os.path.join(data_dir, 'doc2id.json'), 'r') as f:
	doc2id = json.load(f);

class Word2vecDocVectorizer:
	def __init__(self):
		self._word_embeddings = None
		with open(os.path.join(data_dir, 'word2vec_vectors.npy'), 'rb') as f:
			self._word_embeddings = np.load(f);
		self._token_regex = re.compile(r"(?u)\b\w+\b")


	def make_doc_vector(self, text):
		word2vec = self._word_embeddings

		vec = np.zeros(word2vec.shape[1], dtype=np.float64)
		for token in self._token_regex.findall(text):
			if token in token2id:
				vec += word2vec[token2id[token]]
		return vec


	def make_doc_embedding_storage(self, texts):
		text_embeddings = []
		for text in texts:
			text_embeddings.append(self.make_doc_vector(text))

		text_embeddings_arr = np.array(text_embeddings)
		docs_norm = np.linalg.norm(text_embeddings_arr, axis=1).reshape(text_embeddings_arr.shape[0], 1)
		docs_norm = np.where(docs_norm==0, 1, docs_norm)
		unitvec_docs = text_embeddings_arr/docs_norm

		return NumpyEmbeddingStorage(unitvec_docs)


class TfIdfDocVectorizer:
	def __init__(self, vocab_size):
		self._vocab_size = vocab_size
		self._token_regex = re.compile(r"(?u)\b\w+\b")


	def make_doc_vector(self, text):
		vec = np.zeros(self._vocab_size)
		for token in self._token_regex.findall(text):
			if token in token2id:
				vec[token2id[token]] += 1
		return np.log(vec+1)


	def make_doc_embedding_storage(self, texts):
		text_embeddings = []
		i = 0
		text_embeddings_arr = np.zeros((len(texts), self._vocab_size))
		for text in texts:
			#print(i)
			i += 1
			text_embeddings_arr[i-1] = self.make_doc_vector(text)

		docs_norm = np.linalg.norm(text_embeddings_arr, axis=1).reshape(text_embeddings_arr.shape[0], 1)
		docs_norm = np.where(docs_norm==0, 1, docs_norm)
		unitvec_docs = text_embeddings_arr/docs_norm

		return NumpyEmbeddingStorage(unitvec_docs)
	

if datasource == 'cranfield':
	with open('../cranfield_data/cran.json', 'r') as f:
		cran_data = json.load(f)
	with open('../cranfield_data/cran.qrel_full.json', 'r') as f:
		cran_qrel = json.load(f)

	doc_embeds = None
	query_engine = None
	query_vectors = {}
	if cmdargs.vectorizer_type in ['word2vec', 'tfidf']:
		docs = []
		for doc in cran_data:
			docs.append(doc['W'])
		vectorizer = Word2vecDocVectorizer() if cmdargs.vectorizer_type == 'word2vec' else TfIdfDocVectorizer(len(id2freq))
		doc_embeds = vectorizer.make_doc_embedding_storage(docs)
		query_engine = QueryEngineCore(doc_embeds, comparator_func=np.dot)
		for query in cran_qrel:
			query_vec = vectorizer.make_doc_vector(query);
			query_unitvec = query_vec/np.linalg.norm(query_vec)
			query_vectors[query] = query_unitvec
	elif cmdargs.vectorizer_type == 'paragraph2vec':
		para2vec = None
		with open(os.path.join(data_dir, 'paragraph_vectors.npy'), 'rb') as f:
			para2vec = np.load(f)
		docs = []
		for doc in cran_data:
			docs.append(para2vec[doc2id['cran_doc_{}'.format(doc['I'])]])
		doc_embeds = NumpyEmbeddingStorage(np.array(docs))
		def comparator(vec1, vec2):
			sub = np.subtract(vec1,vec2)
			return 15/np.dot(sub, sub)
			#return float(np.dot(vec1, vec2)/np.linalg.norm(vec2))
		query_engine = QueryEngineCore(doc_embeds, comparator_func = comparator)
		for query in cran_qrel:
			if len(cran_qrel[query]) == 0:
				query_vectors[query] = np.zeros(300)
				continue
			query_num = int(cran_qrel[query][0]['qnum'])
			query_vectors[query] = para2vec[doc2id['cran_qry_{}'.format(query_num)]]
		#	print(query, query_num)

	nr = dict()
	for query in cran_qrel:
		query_id = cran_qrel[query][0]['qnum']
		nr[query_id] = []

		doc_rels = {}
		for d in cran_qrel[query]:
			doc_rels[int(d['dnum'])] = int(d['rnum'])

		query_unitvec = query_vectors[query]

		results = query_engine.search(query_unitvec)

		for result in results:
			doc_id = int(cran_data[result[1]]['I'])
			doc_rel = doc_rels[doc_id] if doc_id in doc_rels else 5
			nr[query_id].append({'doc_id': doc_id, 'doc_rel': doc_rel, 'score': result[0]})
			if cmdargs.tsv_output:
				print('{:d}\t{:d}\t{:d}\t{:0.16f}'.format(query_id, doc_id, doc_rel, result[0]))
	#with open('/dev/shm/tmp351.json', 'w') as f:
	#	json.dump(nr, f, indent=4, sort_keys=True)


