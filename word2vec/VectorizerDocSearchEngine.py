
import numpy as np
import json
import sys
import os
import sklearn.svm
import time

import pymongo
import bson.objectid

import itertools

from QueryEngineCore import ComparatorQueryEngineCore, AnnoyQueryEngineCore, TfIdfQueryEngineCore
from DocVectorizer import Word2vecDocVectorizer, TfIdfDocVectorizer, OneShotNetworkDocVectorizer, WordvecAdjustedTfIdfDocVectorizer, ParagraphVectorDocVectorizer
from NumpyEmbeddingStorage import NumpyEmbeddingStorage, SparseEmbeddingStorage, MongoEmbeddingStorage
from MongoInvertedIndex import MongoInvertedIndex


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');


class VectorizerDocSearchEngine:
	def __init__(self, vectorizer_type='tfidf_with_word2vec'):
		self._vectorizer_type = vectorizer_type

		# We expect a MongoDB instance to be running. It should have
		# database 'sharesci', where token2id and id2freq are stored in
		# a 'special_objects' collection
		mongo_client = pymongo.MongoClient('localhost', 27017)
		self._mongo_db = mongo_client['sharesci']

		token2id = self._mongo_db['special_objects'].find_one({'key': 'token2id'})['value']
		self._id2freq = self._mongo_db['special_objects'].find_one({'key': 'id2freq'})['value']

		with open(os.path.join(data_dir, 'word2vec_vectors.npy'), 'rb') as f:
			word_vectors = np.load(f)
		with open(os.path.join(data_dir, 'word2vec_adjacencies1.npy'), 'rb') as f:
			word_adj = np.load(f)

		if self._vectorizer_type == 'word2vec':
			self._vectorizer = Word2vecDocVectorizer(token2id, os.path.join(data_dir, 'word2vec_vectors.npy'))
		elif self._vectorizer_type == 'direct':
			self._vectorizer = OneShotNetworkDocVectorizer(os.path.join(data_dir, 'directdoc2vec_checkpoint.dnn'), filter_size=3)
		elif self._vectorizer_type == 'tfidf_with_word2vec':
			self._vectorizer = WordvecAdjustedTfIdfDocVectorizer(token2id, len(token2id.keys()), word_vectors, word_adj)
		elif self._vectorizer_type == 'paragraph2vec':
			self._vectorizer = ParagraphVectorDocVectorizer('../largedata/para2vec_checkpoint_new.dnn', token2id)
		else:
			self._vectorizer = TfIdfDocVectorizer(token2id, len(token2id.keys()))

		self._inverted_index = MongoInvertedIndex(self._mongo_db['papers_index'], self._mongo_db['special_objects'], self._mongo_db['papers_vecs'], self._get_vec_field_name())

		self._reload_all_docs()


	def _get_vec_field_name(self):
		return 'searchengine_vec_' + self._vectorizer_type


	def _is_vector_type_sparse(self):
		return (self._vectorizer_type == 'tfidf' or self._vectorizer_type == 'tfidf_with_word2vec')


	def _vec_to_sparse_tuples(self, vec):
		nonzeros = vec.nonzero()[0]
		return [(int(x[0]), float(x[1])) for x in zip(nonzeros, vec[nonzeros])]


	def _flush_new_docs(self, new_docs):
		self._inverted_index.index_documents(new_docs)

		# Set a boolean flag to indicate these document vectors is now indexed
		new_doc_ids = [doc[0] for doc in new_docs]
		self._mongo_db['papers'].update({'_id': {'$in': new_doc_ids} }, {'$set': {('other.' + self._get_vec_field_name()): True}}, multi=True)


	def _reload_all_docs(self):
		print('Reloading...')
		start_time = time.perf_counter()

		vec_field_name = self._get_vec_field_name()

		new_docs = []

		num_newly_vectorized = 0
		for mongo_doc in self._mongo_db['papers'].find({('other.' + vec_field_name): {'$exists': False}}, projection={'_id': True, 'abstract': True, 'body': True}):

			text_field = 'body' if 'body' in mongo_doc else 'abstract'
			raw_embed = self._vectorizer.make_doc_vector(mongo_doc[text_field]).astype(float)
			norm_embed = raw_embed / np.sum(np.square(raw_embed))

			if self._is_vector_type_sparse():
				norm_embed = self._vec_to_sparse_tuples(norm_embed)

			new_docs.append((mongo_doc['_id'], norm_embed))
			num_newly_vectorized += 1

			if 2000 < len(new_docs):
				self._flush_new_docs(new_docs)
				new_docs = []

		self._flush_new_docs(new_docs)
		new_docs = []

		self._query_engine = TfIdfQueryEngineCore(self._inverted_index)

		print('Reloaded, with {} new vectors created. Total time: {:0.3f} seconds'.format(num_newly_vectorized, time.perf_counter() - start_time))


	def notify_new_docs(self, new_doc_ids=[]):
		# TODO: Use the new_doc_ids input to update only the docs that
		# changed. Currently this function always reloads everything,
		# which is inefficient and scales poorly
		self._reload_all_docs()


	def search_qs(self, query, **generic_search_kwargs):
		query_vec = self._vectorizer.make_doc_vector(query);
		if self._is_vector_type_sparse():
			query_vec = self._vec_to_sparse_tuples(query_vec)
		return self.search_queryvec(query_vec, **generic_search_kwargs)


	def search_docid(self, doc_id, **generic_search_kwargs):
		query_vec = self._inverted_index.get_doc_vector(bson.objectid.ObjectId(doc_id));
		return self.search_queryvec(query_vec, **generic_search_kwargs)


	def search_userid(self, user_id, max_results=sys.maxsize, offset=0, getFullDocs=False):
		if max_results == 0:
			max_results = sys.maxsize

		# FIXME: Right now, always give empty because the vectors are too big to process.
		if True:
			return []

		## Init the history vector for the user
		#user_history_vec = np.zeros(len(self._doc_embeds))

		## Get user history
		#cursor = self._mongo_db['users'].find({'_id': user_id}, {'_id': 1, 'docIds': 1}).limit(1)
		#if cursor.count() == 0 or 'docIds' not in cursor[0]:
		#	return []

		## Convert user history to vector
		## Note: we make and store the array for doc indexes to use them
		## again later
		#user_docIds_arr = cursor[0]['docIds']
		#user_docIdxs = set()
		#for docId in user_docIds_arr:
		#	if docId not in self._id2idx:
		#		continue
		#	user_docIdxs.add(self._id2idx[docId])
		#for docIdx in user_docIdxs:
		#	user_history_vec[docIdx] = 1

		## If there weren't any example docs, the SVM can't train, so
		## just fail and return empty
		#if len(user_docIdxs) == 0:
		#	return []

		## Train the SVM
		## TODO: Move this process offline so we don't have to train from scratch on every query
		#user_svm = sklearn.svm.LinearSVC(class_weight='balanced', verbose=False, max_iter = 2000, C=0.1, tol=1e-5)
		#user_svm.fit(self._doc_embeds.as_numpy_array(), user_history_vec)

		#results = ComparatorQueryEngineCore.search_static(np.zeros(1), self._doc_embeds, lambda vec, qvec: float(user_svm.decision_function([vec])), max_results=(offset+max_results+len(user_docIds_arr)))

		## Filter to exclude any docs the user has already seen
		#results[:] = itertools.filterfalse((lambda x: x[1] in user_docIdxs), results)

		## Get the specified result window
		#results = results[offset:offset+max_results]

		#return self._convert_result_format(results)
		

	def search_queryvec(self, query_vec, max_results=sys.maxsize, offset=0, getFullDocs=False):
		if max_results == 0:
			max_results = sys.maxsize

		results = self._query_engine.search(query_vec, max_results=(offset+max_results))[offset:(offset+max_results)]

		return results

