
import numpy as np
import json
import sys
import os
from QueryEngineCore import QueryEngineCore
from DocVectorizer import Word2vecDocVectorizer, TfIdfDocVectorizer


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');
cranfield_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cranfield_data');

class VectorizerDocSearchEngine:
	def __init__(self):
		with open(os.path.join(data_dir, 'token2id.json'), 'r') as f:
			token2id = json.load(f);

		with open(os.path.join(data_dir, 'id2freq.npy'), 'rb') as f:
			self._id2freq = np.load(f);

		with open(os.path.join(cranfield_dir, 'cran.json'), 'r') as f:
			cran_data = json.load(f)

		docs = []
		self._idx2id = []
		self._id2idx = dict()
		for doc in cran_data:
			# ID is mapped to the current index in the `docs` array
			# Obviously, it's important to do this before appending
			# to `docs`.
			self._id2idx[doc['I']] = len(docs)

			# Map the current doc's index in the array to its ID
			self._idx2id.append(doc['I'])

			docs.append(doc['W'])


		#self._vectorizer = TfIdfDocVectorizer(token2id, len(id2freq))
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
			res_aslist[1] = int(self._idx2id[result[1]])
			converted_results.append(tuple(res_aslist))

		return converted_results
