
import numpy as np
import json
import sys
import os

import pymongo

from QueryEngineCore import QueryEngineCore
from DocVectorizer import Word2vecDocVectorizer, TfIdfDocVectorizer


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');

class VectorizerDocSearchEngine:
	def __init__(self):
		# We expect a MongoDB instance to be running. It should have
		# database 'sharesci', where token2id and id2freq are stored in
		# a 'special_objects' collection and documents are from the
		# 'cranfield' collection.
		mongo_client = pymongo.MongoClient('localhost', 27017)
		self._mongo_db = mongo_client['sharesci']

		token2id = self._mongo_db['special_objects'].find_one({'key': 'token2id'})['value']
		self._id2freq = self._mongo_db['special_objects'].find_one({'key': 'id2freq'})['value']

		docs = []
		self._idx2id = []
		self._id2idx = dict()
		for mongo_doc in self._mongo_db['cranfield'].find({}):
			# ID is mapped to the current index in the `docs` array
			# Obviously, it's important to do this before appending
			# to `docs`.
			self._id2idx[str(mongo_doc['_id'])] = len(docs)

			# Map the current doc's index in the array to its ID
			self._idx2id.append(str(mongo_doc['_id']))

			docs.append(mongo_doc['body'])


		self._vectorizer = Word2vecDocVectorizer(token2id, os.path.join(data_dir, 'word2vec_vectors.npy'))

		self._doc_embeds = self._vectorizer.make_doc_embedding_storage(docs)
		self._query_engine = QueryEngineCore(self._doc_embeds, comparator_func=np.dot)


	def search_qs(self, query, **generic_search_kwargs):
		query_vec = self._vectorizer.make_doc_vector(query);
		query_unitvec = query_vec/np.linalg.norm(query_vec)

		return self.search_queryvec(query_unitvec, **generic_search_kwargs)


	def search_docid(self, doc_id, **generic_search_kwargs):
		query_vec = self._doc_embeds.get_by_id(self._id2idx[doc_id]);
		return self.search_queryvec(query_vec, **generic_search_kwargs)


	def search_queryvec(self, query_vec, max_results=sys.maxsize, offset=0, getFullDocs=False):
		if max_results == 0:
			max_results = sys.maxsize

		results = self._query_engine.search(query_vec)[offset:(offset+max_results)]

		# Convert doc IDs
		converted_results = []
		for result in results:
			res_aslist = list(result)
			res_aslist[1] = self._idx2id[result[1]]
			converted_results.append(tuple(res_aslist))

		return converted_results
