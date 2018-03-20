#!/usr/bin/env -p python3

import heapq
import numpy as np
import sys
import annoy

class ComparatorQueryEngineCore:
	def __init__(self, embedding_storage, comparator_func):
		self._embeddings = embedding_storage
		self._comparator_func = comparator_func


	def search(self, query_vector, **kwargs):
		return ComparatorQueryEngineCore.search_static(query_vector, self._embeddings, self._comparator_func, **kwargs)


	def search_static(query_vector, embeddings, comparator, max_results=sys.maxsize):
		if len(query_vector) != embeddings.embedding_size():
			query_vector = np.resize(query_vector, embeddings.embedding_size())

		num_results = min(max_results, len(embeddings))

		# Init the heap to a fixed size for efficient k-max
		scores_heap = [(-sys.maxsize,)] * num_results

		for doc_num in range(len(embeddings)):
			vec = embeddings.get_by_id(doc_num)
			# Note this is a min-heap, so top will be the WORST
			# match. We push the current score, then pop the top to
			# get rid of the worst match
			heapq.heappushpop(scores_heap, (comparator(vec, query_vector), doc_num))

		# Note again that this is a min-heap, so initially the array
		# will be sorted in WORST first order for the top results, so
		# needs to be reversed
		results = [heapq.heappop(scores_heap) for i in range(num_results)]
		results.reverse()
		print(results)

		return results

class AnnoyQueryEngineCore:
	def __init__(self, embedding_storage, annoy_index=None):
		self._embeddings = embedding_storage
		self._annoy_index = annoy_index
		if self._annoy_index is None:
			self._annoy_index = annoy.AnnoyIndex(self._embeddings.embedding_size(), metric='euclidean')

		self._build_index()


	def _build_index(self):
		for i in range(len(self._embeddings)):
			self._annoy_index.add_item(i, self._embeddings.get_by_id(i))
		self._annoy_index.build(2 * self._embeddings.embedding_size())


	def search(self, query_vector, **kwargs):
		return AnnoyQueryEngineCore.search_static(query_vector, self._annoy_index, **kwargs)


	def search_static(query_vector, annoy_index, max_results=sys.maxsize):
		annoy_results = annoy_index.get_nns_by_vector(query_vector, min(max_results, annoy_index.get_n_items()), include_distances=True)
		results = []
		for i in range(len(annoy_results[0])):
			# Negate annoy_results[1] to convert distance (where
			# lower is better) to score (where higher is better)
			results.append((-annoy_results[1][i], annoy_results[0][i]))

		return results

