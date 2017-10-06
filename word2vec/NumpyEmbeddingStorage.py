#!/usr/bin/env -p python3

## Wrapper around a numpy array of document or word embeddings.
#
# Why use this instead of just using the array directly? This class has a more
# modular interface that could be implemented to wrap other storage methods as
# well. For example, a storage that keeps data on disk and out of RAM as much
# as possible (i.e., to reduce RAM usage).
#
class NumpyEmbeddingStorage:
	def __init__(self, embeddings_arr):
		self._embeddings_arr = embeddings_arr


	def as_numpy_array(self):
		return self._embeddings_arr


	def get_by_id(self, doc_id):
		return self._embeddings_arr[doc_id]


	def embedding_size(self):
		return self._embeddings_arr.shape[1]


	def __len__(self):
		return self._embeddings_arr.shape[0]


	def __iter__(self):
		return EmbeddingStorageIter(self)


	def __get__(self, i):
		return self._embeddings_arr[i]


class EmbeddingStorageIter:
	def __init__(self, embedding_storage):
		self._storage = embedding_storage
		self._index = -1
		self._storage_len = len(embedding_storage)


	def __iter__(self):
		return self


	def next(self):
		if self._index == self._storage_len:
			raise StopIteration()

		self._index += 1
		return self._storage[self._index]
