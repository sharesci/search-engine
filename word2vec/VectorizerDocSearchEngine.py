
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
		for doc in cran_data:
			docs.append(doc['W'])
			self._idx2id.append(doc['I'])

		#self._vectorizer = TfIdfDocVectorizer(token2id, len(id2freq))
		self._vectorizer = Word2vecDocVectorizer(token2id, os.path.join(data_dir, 'word2vec_vectors.npy'))

		doc_embeds = self._vectorizer.make_doc_embedding_storage(docs)
		self._query_engine = QueryEngineCore(doc_embeds, comparator_func=np.dot)


	def search_qs(self, query, max_results=sys.maxsize, offset=0, getFullDocs=False):
		if max_results == 0:
			max_results = sys.maxsize

		query_vec = self._vectorizer.make_doc_vector(query);
		query_unitvec = query_vec/np.linalg.norm(query_vec)

		results = self._query_engine.search(query_unitvec)[offset:(offset+max_results)]

		# Convert doc IDs
		converted_results = []
		for result in results:
			res_aslist = list(result)
			res_aslist[1] = int(self._idx2id[result[1]])
			converted_results.append(tuple(res_aslist))

		return converted_results
			

