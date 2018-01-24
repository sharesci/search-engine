
import numpy as np
import re
from NumpyEmbeddingStorage import NumpyEmbeddingStorage


class DocVectorizer:
	def __init__(self):
		pass


	## Makes a single doc vector from the given string.
	#
	# @param text (string)
	# <br>  The text to vectorize
	# 
	# @return (numpy array)
	# <br>  A vector created from the text
	#
	def make_doc_vector(self, text):
		pass


	## Makes a single doc vector from the given string.
	#
	# @param texts (list of string)
	# <br>  A list of N texts to vectorize.
	# 
	# @return (EmbeddingStorage object)
	#
	def make_doc_embedding_storage(self, texts):
		pass


class Word2vecDocVectorizer(DocVectorizer):
	def __init__(self, token2id, vec_file_path):
		super().__init__();

		self._token2id = token2id

		self._word_embeddings = None
		with open(vec_file_path, 'rb') as f:
			self._word_embeddings = np.load(f);
		self._token_regex = re.compile(r"(?u)\b\w+\b")


	def make_doc_vector(self, text):
		word2vec = self._word_embeddings

		vec = np.zeros(word2vec.shape[1], dtype=np.float64)
		for token in self._token_regex.findall(text):
			if token in self._token2id:
				vec += word2vec[self._token2id[token]]
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


class TfIdfDocVectorizer(DocVectorizer):
	def __init__(self, token2id, vocab_size):
		super().__init__();

		self._token2id = token2id

		self._vocab_size = vocab_size
		self._token_regex = re.compile(r"(?u)\b\w+\b")


	def make_doc_vector(self, text):
		vec = np.zeros(self._vocab_size)
		for token in self._token_regex.findall(text):
			if token in self._token2id:
				vec[self._token2id[token]] += 1
		return np.log(vec+1)


	def make_doc_embedding_storage(self, texts):
		text_embeddings = []
		i = 0
		text_embeddings_arr = np.zeros((len(texts), self._vocab_size))
		for text in texts:
			i += 1
			text_embeddings_arr[i-1] = self.make_doc_vector(text)

		docs_norm = np.linalg.norm(text_embeddings_arr, axis=1).reshape(text_embeddings_arr.shape[0], 1)
		docs_norm = np.where(docs_norm==0, 1, docs_norm)
		unitvec_docs = text_embeddings_arr/docs_norm

		return NumpyEmbeddingStorage(unitvec_docs)

