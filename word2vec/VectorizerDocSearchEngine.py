
import numpy as np
import json
import sys
import os
import sklearn.svm
import time
import heapq
import re

import pymongo
import bson.objectid

import itertools

from QueryEngineCore import ComparatorQueryEngineCore, AnnoyQueryEngineCore, TfIdfQueryEngineCore
from DocVectorizer import Word2vecDocVectorizer, TfIdfDocVectorizer, OneShotNetworkDocVectorizer, WordvecAdjustedTfIdfDocVectorizer, ParagraphVectorDocVectorizer
from NumpyEmbeddingStorage import NumpyEmbeddingStorage, SparseEmbeddingStorage, MongoEmbeddingStorage
from MongoInvertedIndex import MongoInvertedIndex


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');


class VectorizerDocSearchEngine:
	def __init__(self):

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

		self._tfidf_vec_field_name = 'searchengine_vec_tfidf_with_word2vec'
		self._tfidf_vectorizer = WordvecAdjustedTfIdfDocVectorizer(token2id, len(token2id.keys()), word_vectors, word_adj)

		self._paragraph_vec_field_name = 'searchengine_vec_paragraph2vec'
		self._paragraph_vectorizer = ParagraphVectorDocVectorizer('../largedata/para2vec_checkpoint_new.dnn', token2id)

		self._inverted_index = MongoInvertedIndex(self._mongo_db['papers_index'], self._mongo_db['special_objects'], self._mongo_db['papers_vecs'], self._tfidf_vec_field_name)

		self._query_engine = TfIdfQueryEngineCore(self._inverted_index)

		self._cache = dict()
		self._max_cache_age = 7200   # seconds
		self._cache_access_counter = 0

		self._reload_all_docs()


	def _garbage_collect_caches(self):
		cur_time = time.time()
		for main_key in self._cache:
			to_remove = []
			for entry in self._cache[main_key]:
				if entry['expire_time'] < cur_time:
					to_remove.append(entry)
			# Shortcut when removing all entries
			if len(to_remove) == len(self._cache[main_key]):
				del self._cache[main_key]
				continue

			for entry in to_remove:
				self._cache[main_key].remove(entry)

	
	## Returns a result set if a cached one is found for the given
	# parameters. If there is not a cache hit, returns None.
	#
	def _get_results_from_cache(self, main_key, extra_keys):
		if main_key not in self._cache:
			return None

		cache_entries = self._cache[main_key]
		for entry in cache_entries:
			# We intentionally only check against the given extra
			# keys here, even if the cache entry itself had more
			# extra keys
			is_match = True
			for key in extra_keys:
				if key not in entry['extra_keys'] or entry['extra_keys'][key] != extra_keys[key]:
					is_match = False
					break
			if not is_match:
				continue

			if entry['expire_time'] <= time.time():
				cache_entries.remove(entry)
				break

			return entry['value']

		return None


	def _add_to_cache(self, main_key, extra_keys, value):
		self._cache_access_counter += 1
		if 20 < self._cache_access_counter:
			self._garbage_collect_caches()

		if main_key not in self._cache:
			self._cache[main_key] = []

		self._cache[main_key].append({
			'expire_time': time.time() + self._max_cache_age,
			'extra_keys': extra_keys,
			'value': value
		});


	def _vec_to_sparse_tuples(self, vec):
		nonzeros = vec.nonzero()[0]
		return [(int(x[0]), float(x[1])) for x in zip(nonzeros, vec[nonzeros])]


	def _flush_new_tfidf_docs(self, new_tfidf_docs):
		if len(new_tfidf_docs) == 0:
			return
		self._inverted_index.index_documents(new_tfidf_docs)

		# Set a boolean flag to indicate these document vectors is now indexed
		new_doc_ids = [doc[0] for doc in new_tfidf_docs]
		self._mongo_db['papers'].update({'_id': {'$in': new_doc_ids} }, {'$set': {('other.' + self._tfidf_vec_field_name): True}}, multi=True)


	def _flush_new_para2vec_docs(self, new_para2vec_docs):
		if len(new_para2vec_docs) == 0:
			return
		texts = [doc[1] for doc in new_para2vec_docs]
		vectors = self._paragraph_vectorizer.make_doc_vectors(texts).astype(float)
		for i in range(len(new_para2vec_docs)):
			self._mongo_db['papers'].update({'_id': new_para2vec_docs[i][0] }, {'$set': {('other.' + self._paragraph_vec_field_name): list(vectors[i])}})


	def _reload_all_docs(self):
		print('Reloading...')
		start_time = time.perf_counter()

		new_tfidf_docs = []
		new_para2vec_docs = []

		num_newly_vectorized = {'paragraph2vec': 0, 'tfidf': 0}
		doc_query = {'$or': [
			{('other.' + self._tfidf_vec_field_name): {'$exists': False}},
			{('other.' + self._paragraph_vec_field_name): {'$exists': False}}
		]}
		doc_projection = {'_id': True,
			'abstract': True,
			'body': True,
			('other.' + self._tfidf_vec_field_name): True,
			('other.' + self._paragraph_vec_field_name): True
		}
		mongo_cursor = self._mongo_db['papers'].find(doc_query, projection=doc_projection)
		mongo_cursor.batch_size(400);
		for mongo_doc in mongo_cursor:

			if 'other' not in mongo_doc or self._tfidf_vec_field_name not in mongo_doc['other']:
				text_field = 'body' if 'body' in mongo_doc else 'abstract'
				raw_embed = self._tfidf_vectorizer.make_doc_vector(mongo_doc[text_field]).astype(float)
				embed_len = np.sum(np.square(raw_embed))

				# Prevent divide-by-zero errors
				if embed_len == 0:
					embed_len = 1

				norm_embed = self._vec_to_sparse_tuples(raw_embed / embed_len)

				new_tfidf_docs.append((mongo_doc['_id'], norm_embed))
				num_newly_vectorized['tfidf'] += 1

			if 'other' not in mongo_doc or self._paragraph_vec_field_name not in mongo_doc['other']:
				text_field = 'body' if 'body' in mongo_doc else 'abstract'
				new_para2vec_docs.append((mongo_doc['_id'], mongo_doc[text_field]))
				num_newly_vectorized['paragraph2vec'] += 1

			if 900 < len(new_para2vec_docs):
				self._flush_new_para2vec_docs(new_para2vec_docs)
				new_para2vec_docs = []
			if 20000 <= len(new_tfidf_docs):
				self._flush_new_tfidf_docs(new_tfidf_docs)
				new_tfidf_docs = []

		self._flush_new_para2vec_docs(new_para2vec_docs)
		self._flush_new_tfidf_docs(new_tfidf_docs)

		print('Reloaded: {} new paragraph vectors, {} new TF-IDFs. Total time: {:0.3f} seconds'.format(num_newly_vectorized['paragraph2vec'], num_newly_vectorized['tfidf'], time.perf_counter() - start_time))


	def notify_new_docs(self, new_doc_ids=[]):
		# TODO: Use the new_doc_ids input to update only the docs that
		# changed. Currently this function always reloads everything,
		# which is inefficient and scales poorly
		self._reload_all_docs()


	def search_qs(self, query, **generic_search_kwargs):
		query = re.sub(r"\s+", " ", query.lower())

		cache_extra_keys = {'max_results': 10, 'offset': 0}
		if 'max_results' in generic_search_kwargs:
			cache_extra_keys['max_results'] = generic_search_kwargs['max_results']
		if 'offset' in generic_search_kwargs:
			cache_extra_keys['offset'] = generic_search_kwargs['offset']
		main_key = 'search_qs.' + query
		cached_result = self._get_results_from_cache(main_key, cache_extra_keys)
		if cached_result is not None:
			return cached_result

		query_vec = self._tfidf_vectorizer.make_doc_vector(query);
		query_vec = self._vec_to_sparse_tuples(query_vec)
		results = self.search_queryvec(query_vec, **generic_search_kwargs)

		self._add_to_cache(main_key, cache_extra_keys, results)

		return results


	def search_docid(self, doc_id, max_results=20, offset=0, **generic_search_kwargs):
		if not isinstance(doc_id, bson.objectid.ObjectId):
			doc_id = bson.objectid.ObjectId(doc_id)

		cache_extra_keys = {'max_results': max_results, 'offset': offset}
		main_key = 'search_docid.' + str(doc_id)
		cached_result = self._get_results_from_cache(main_key, cache_extra_keys)
		if cached_result is not None:
			return cached_result

		# Start with a TF-IDF search to get a rough set of candidates
		qs_max_results = max_results * 2 + offset
		querydoc_tfvec = self._inverted_index.get_doc_vector(doc_id);
		base_results = self._query_engine.search(querydoc_tfvec, certainty_factor=0.5, max_results=qs_max_results)

		# Now re-rank the candidates based on paragraph vector nearest neighbor
		mongo_querydoc = self._mongo_db['papers'].find_one({'_id': doc_id, ('other.' + self._paragraph_vec_field_name): {'$exists': True}}, {'_id': True, ('other.' + self._paragraph_vec_field_name): True})

		# Paragraph vectors can take time to generate, so if this doc
		# doesn't have one we'll just give back the TF-IDF result
		if mongo_querydoc is None:
			if len(base_results) < offset+1:
				return []
			return base_results[offset:offset+max_results]

		querydoc_paravec = np.array(mongo_querydoc['other'][self._paragraph_vec_field_name])

		result_doc_ids = [bson.objectid.ObjectId(doc[1]) for doc in base_results]
		result_doc_cursor = self._mongo_db['papers'].find({'_id': {'$in': result_doc_ids}, ('other.' + self._paragraph_vec_field_name): {'$exists': True}}, {'_id': True, ('other.' + self._paragraph_vec_field_name): True})

		final_results = []

		for result_doc in result_doc_cursor:
			doc_vec_diff = np.subtract(result_doc['other'][self._paragraph_vec_field_name], querydoc_paravec)
			doc_score = -np.dot(doc_vec_diff, doc_vec_diff)
			final_results.append((doc_score, str(result_doc['_id'])))

		final_results = sorted(final_results, key=lambda x: x[0], reverse=True)

		if len(final_results) < offset+1:
			return []

		results = final_results[offset:offset+max_results]
		self._add_to_cache(main_key, cache_extra_keys, results)

		return results


	def search_userid(self, user_id, max_results=sys.maxsize, offset=0, getFullDocs=False, history_window=100):
		if max_results == 0:
			max_results = sys.maxsize

		cache_extra_keys = {'max_results': max_results, 'offset': offset}
		main_key = 'search_userid.' + str(user_id)
		cached_result = self._get_results_from_cache(main_key, cache_extra_keys)
		if cached_result is not None:
			return cached_result

		# Get user history
		user_blob = self._mongo_db['users'].find_one({'_id': user_id}, {'_id': 1, 'docIds': 1, 'terms': 1})

		# Can't do anything if the user doesn't exist or doesn't have history
		if user_blob is None or (('docIds' not in user_blob or len(user_blob['docIds']) == 0) and ('terms' not in user_blob or len(user_blob['terms']) == 0)):
			return []

		# Two possibilities to construct the initial list of
		# candidates: (1) if we have search terms, do a quick TF-IDF on
		# those; (2) if we don't have search terms, do a related-doc
		# search on the docId history
		candidate_docs = []
		if 'terms' in user_blob and len(user_blob['terms']) != 0:
			recent_terms = ' '.join(user_blob['terms'][:history_window])
			qs_max_results = max_results * 2 + offset
			candidate_docs = self.search_qs(recent_terms, offset=0, max_results=qs_max_results, certainty_factor=0.05)

			# We would need doc IDs to train the SVM. If we don't
			# have any, just return the candidates
			if 'docIds' not in user_blob or len(user_blob['docIds']) == 0:
				if len(candidate_docs) < offset:
					return []
				return candidate_docs[offset:offset+max_results]
		elif 'docIds' in user_blob and len(user_blob['docIds']) != 0:
			# It would take a very long time to do related-doc
			# searches over the entire history window, so we'll
			# just do a random sample
			recent_docIds = random.sample(user_blob['docIds'][:history_window], 5)
			candidate_docs = []
			for recent_docId in recent_docIds:
				candidate_docs += self.search_docid(bson.objectid.ObjectId(recent_docId), offset=0, max_results=max_results)

		history_doc_ids = [bson.objectid.ObjectId(doc_id) for doc_id in user_blob['docIds']]
		recent_history_doc_ids = history_doc_ids[:history_window]

		# Get the paragraph vectors for the candidate docs
		candidate_doc_vecs = []
		candidate_doc_ids = [bson.objectid.ObjectId(doc[1]) for doc in candidate_docs]
		for mongo_doc in self._mongo_db['papers'].find({'_id': {'$in': candidate_doc_ids}, ('other.' + self._paragraph_vec_field_name): {'$exists': True}}, projection={'_id': True, ('other.' + self._paragraph_vec_field_name): True}):
			candidate_doc_vecs.append((mongo_doc['_id'], mongo_doc['other'][self._paragraph_vec_field_name]))

		history_doc_vecs = []
		for mongo_doc in self._mongo_db['papers'].find({'_id': {'$in': recent_history_doc_ids}, ('other.' + self._paragraph_vec_field_name): {'$exists': True}}, projection={'_id': True, ('other.' + self._paragraph_vec_field_name): True}):
			history_doc_vecs.append((mongo_doc['_id'], mongo_doc['other'][self._paragraph_vec_field_name]))

		# Maybe the user's history contains only expired documents
		# (that are no longer in the database)
		if len(history_doc_vecs) == 0:
			if len(candidate_docs) < offset:
				return []
			return candidate_docs[offset:offset+max_results]

		# Get some random docs not in the history as negative examples for the SVM
		excluded_doc_ids = candidate_doc_ids + recent_history_doc_ids
		random_doc_query = {
			'_id': {'$not': {'$in': excluded_doc_ids}},
			('other.' + self._paragraph_vec_field_name): {'$exists': True}
		}
		random_doc_cursor = self._mongo_db['papers'].aggregate([{'$match': random_doc_query}, {'$sample': {'size': 100}}, {'$project': {'_id': True, ('other.' + self._paragraph_vec_field_name): True}}])
		random_docs = []
		for doc in random_doc_cursor:
			random_docs.append((doc['_id'], doc['other'][self._paragraph_vec_field_name]))

		# Put everything into a dict. Note that this is done in such
		# a way that duplicates will be eliminated (they will just
		# overwrite the same dict key)
		all_training_docs = dict()
		for doc in random_docs:
			all_training_docs[doc[0]] = doc[1]
		for doc in candidate_doc_vecs:
			all_training_docs[doc[0]] = doc[1]
		for doc in history_doc_vecs:
			all_training_docs[doc[0]] = doc[1]

		# Init training data matrix
		embed_size = len(random_docs[0][1])
		num_training_docs = len(all_training_docs)
		training_matrix = np.zeros((num_training_docs, embed_size))

		# Init the history vector for the user
		user_history_vec = np.zeros(num_training_docs)

		id2idx = dict()
		idx2id = dict()
		i = 0
		for doc_id in all_training_docs.keys():
			training_matrix[i][:] = np.array(all_training_docs[doc_id])
			if bson.objectid.ObjectId(doc_id) in history_doc_ids:
				user_history_vec[i] = 1
			id2idx[doc_id] = i
			idx2id[i] = doc_id
			i += 1

		# Train the SVM
		# TODO: Move this process offline so we don't have to train from scratch on every query
		user_svm = sklearn.svm.LinearSVC(class_weight='balanced', verbose=False, max_iter = 2000, C=0.1, tol=1e-5)
		user_svm.fit(training_matrix, user_history_vec)

		results = ComparatorQueryEngineCore.search_static(np.zeros(1), NumpyEmbeddingStorage(training_matrix), lambda vec, qvec: float(user_svm.decision_function([vec])), max_results=(offset+max_results+len(recent_history_doc_ids)))

		# Convert the indexes back into doc IDs
		results = [(score, idx2id[idx]) for score, idx in results]

		# Filter to exclude any docs the user has already seen
		results[:] = itertools.filterfalse((lambda x: x[1] in history_doc_ids), results)

		# Get the specified result window
		results = results[offset:offset+max_results]

		# Convert the doc IDs into strings
		results = [(score, str(doc_id)) for score, doc_id in results]
		self._add_to_cache(main_key, cache_extra_keys, results)

		return results


	def search_queryvec(self, query_vec, max_results=sys.maxsize, offset=0, getFullDocs=False, **kwargs):
		if max_results == 0:
			max_results = sys.maxsize

		results = self._query_engine.search(query_vec, max_results=(offset+max_results), **kwargs)[offset:(offset+max_results)]

		return results

