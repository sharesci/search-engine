#!/usr/bin/env -p python3

import heapq
import numpy as np
import sys
import annoy
import random

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

		for doc_id in embeddings.id_iter():
			vec = embeddings.get_by_id(doc_id)
			# Note this is a min-heap, so top will be the WORST
			# match. We push the current score, then pop the top to
			# get rid of the worst match
			heapq.heappushpop(scores_heap, (comparator(vec, query_vector), doc_id))

		# Note again that this is a min-heap, so initially the array
		# will be sorted in WORST first order for the top results, so
		# needs to be reversed
		results = [heapq.heappop(scores_heap) for i in range(num_results)]
		results.reverse()

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


class TfIdfQueryEngineCore:
	def __init__(self, inverted_index):
		self._inverted_index = inverted_index


	def search(self, query_vector, **kwargs):
		return TfIdfQueryEngineCore.search_static(query_vector, self._inverted_index, **kwargs)


	def _calc_max_possible(term_weights, term_curvals):
		max_possible = 0
		for term in term_weights.keys():
			max_possible += term_weights[term] * term_curvals[term]

		return max_possible


	def search_static(query_vector, inverted_index, max_results=sys.maxsize, search_alpha=0.9):
		query_dict = {v[0]: v[1] for v in query_vector}
		term_infos = {term_id: inverted_index.get_term_info(term_id) for term_id in query_dict.keys()}
		num_docs = inverted_index.get_num_docs()

		term_weights = dict()
		term_iterators = dict()
		term_curvals = dict()
		term_gradients = dict()
		available_term_ids = list(term_id for term_id in query_dict.keys())
		for term_id in available_term_ids:
			if term_infos[term_id] is None or 'df' not in term_infos[term_id]:
				term_weights[term_id] = 0
			else:
				term_weights[term_id] = np.log(num_docs / (term_infos[term_id]['df'] + 1)) * query_dict[term_id]
			term_iterators[term_id] = inverted_index.get_term_iterator(term_id)
			term_curvals[term_id] = 1

		for term_id in available_term_ids:
			term_gradients[term_id] = term_weights[term_id] * 0.01

		max_possible = TfIdfQueryEngineCore._calc_max_possible(term_weights, term_curvals)
		num_results = min(max_results, num_docs)

		# Init the heap to a fixed size for efficient k-max
		scores_heap = [(-sys.maxsize, None)] * num_results

		scored_doc_ids = set()

		while scores_heap[0][0] < max_possible and len(available_term_ids) != 0:
			# Pick the next term to advance
			if random.random() < 0.6:
				cur_term = random.choice(available_term_ids)
			else:
				cur_term = max(available_term_ids, key=lambda x: term_gradients[x])

			doc_id = None
			val = None
			try:
				doc_id, val = next(term_iterators[cur_term])
				term_curvals[cur_term] = val
			except StopIteration:
				available_term_ids.remove(cur_term)
				term_curvals[cur_term] = 0

				old_max_possible = max_possible
				max_possible = TfIdfQueryEngineCore._calc_max_possible(term_weights, term_curvals)
				term_gradients[term_id] = (1 - search_alpha) * term_gradients[term_id] + search_alpha * (old_max_possible - max_possible)

				continue

			old_max_possible = max_possible
			max_possible = TfIdfQueryEngineCore._calc_max_possible(term_weights, term_curvals)
			term_gradients[term_id] = (1 - search_alpha) * term_gradients[term_id] + search_alpha * (old_max_possible - max_possible)

			if doc_id in scored_doc_ids:
				continue

			# Compute score for uncovered doc via cosine similarity
			doc_vector = inverted_index.get_doc_vector(doc_id)
			doc_score = sum(term_weights[v[0]] * v[1] for v in doc_vector if v[0] in term_weights)

			scored_doc_ids.add(doc_id)

			# Note this is a min-heap, so top will be the WORST
			# match. We push the current score, then pop the top to
			# get rid of the worst match
			heapq.heappushpop(scores_heap, (doc_score, doc_id))

		# Get the result array
		results = []
		for i in range(num_results):
			next_result = heapq.heappop(scores_heap)

			# Skip the result if it was just a placeholder
			# (shouldn't happen, but just in case...)
			if next_result[1] is None:
				continue

			results.append((next_result[0], str(next_result[1])))

		# Note again that we used a min-heap, so initially the array
		# will be sorted in WORST first order for the top results, so
		# needs to be reversed
		results.reverse()

		return results

