#!/usr/bin/env -p python3

import heapq
import numpy as np
import sys

class QueryEngineCore:
	def __init__(self, embedding_storage, comparator_func):
		self._embeddings = embedding_storage
		self._comparator_func = comparator_func


	def search(self, query_vector, max_results=sys.maxsize):
		if len(query_vector) != self._embeddings.embedding_size():
			query_vector.resize(self._embeddings.embedding_size())

		num_results = min(max_results, len(self._embeddings))

		# Init the heap to a fixed size for efficient k-max
		scores_heap = [(-sys.maxsize,)] * num_results

		for doc_num in range(len(self._embeddings)):
			vec = self._embeddings.get_by_id(doc_num)
			# Note this is a min-heap, so top will be the WORST
			# match. We push the current score, then pop the top to
			# get rid of the worst match
			heapq.heappushpop(scores_heap, (self._comparator_func(vec, query_vector), doc_num))

		# Note again that this is a min-heap, so initially the array
		# will be sorted in WORST first order for the top results, so
		# needs to be reversed
		results = [heapq.heappop(scores_heap) for i in range(num_results)]
		results.reverse()

		return results

