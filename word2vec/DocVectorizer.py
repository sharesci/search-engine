
import numpy as np
import cntk as C
import re
from NumpyEmbeddingStorage import NumpyEmbeddingStorage

C.try_set_default_device(C.gpu(0))


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


class OneShotNetworkDocVectorizer(DocVectorizer):
	def __init__(self, network_file_path, filter_size=3):
		super().__init__();

		self._filter_size = filter_size

		self._network = C.ops.combine(C.load_model(network_file_path).outputs[0])
		self._embed_dim = self._network.outputs[0].shape[0]
		self._net_input_var = self._network.arguments[0]


	def _str_to_inputs(self, text_str):
		full_arr = np.zeros(len(text_str)+self._filter_size-1, dtype=np.float32)
		for i in range(len(text_str)):
			full_arr[i] = ord(text_str[i])

		ret_data = np.zeros((len(text_str)+self._filter_size-1, self._filter_size), dtype=np.float32)
		for i in range(len(text_str)):
			ret_data[i+self._filter_size-1] = full_arr[i:i+self._filter_size]
		return ret_data


	def make_doc_vector(self, text):
		inputs = [self._str_to_inputs(text)]

		# Split into batches of 50,000 so CNTK does not try to allocate
		# GPU memory for the whole unrolled sequence at once
		vec = self._network.eval(({self._net_input_var: [inputs[0][0:50000]]}, [True]))[0]
		for i in range(1, (len(inputs[0])-1)//50000+1):
			vec = self._network.eval(({self._net_input_var: [inputs[0][i*50000:min((i+1)*50000, len(inputs[0]))]]}, [True]))[0]
			#vec = self._network.eval(({self._net_input_var: [inputs[0][i*50000:(i+1)*50000]]}, [False]))[0]
		return vec


	def make_doc_embedding_storage(self, texts):
		text_embeddings = []
		i = 0
		text_embeddings_arr = np.zeros((len(texts), self._embed_dim))
		for text in texts:
			i += 1
			text_embeddings_arr[i-1][:] = self.make_doc_vector(text)

		return NumpyEmbeddingStorage(text_embeddings_arr)


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



class WordvecAdjustedTfIdfDocVectorizer(TfIdfDocVectorizer):
	##
	# Like regular TF-IDF, but the TF vectors are adjusted based on the
	# similarity of words. For example, since "walk" and "walking" are
	# similar words and therefore have similar word2vec representations,
	# when this method encounters the word "walk" it might add 1 to the
	# "walk" component of the TF vector just like regular TF-IDF, but
	# unlike normal TF-IDF it might also add 0.66 to the "walking"
	# component. The exact amount added to each componenet is based on the
	# distance of the word vectors.
	#
	# @param word_embeddings
	# The table of word vectors (mapped by token2id)
	#
	# @param word_neighbors
	# Table (2D numpy array) of the closest words to each word, for performance
	# reasons. Like word_embeddings, each element on the 0 axis represents
	# a word. The 1 axis is a list of indices of the closest words, in
	# order of increasing distance.
	#
	# @param nearest_k
	# How many nearest neighbors to involve when augmenting the TF vectors.
	# Note that nearest_k=1 is equivalent to regular TF-IDF (just add
	# a 1 to the componenet of the exact word found), nearest_k=2 puts
	# a 1 at the exact word and affects one similar word, and so on.
	# If word_neighbors does not have nearest_k neighbors listed for all
	# words, nearest_k will be reduced to the largest number of neighbors
	# available in word_neighbors.
	#
	def __init__(self, token2id, vocab_size, word_embeddings, word_neighbors, nearest_k=5):
		super().__init__(token2id, vocab_size);

		self._token2id = token2id
		self._vocab_size = vocab_size
		self._word_embeddings = word_embeddings
		self._word_neighbors = word_neighbors
		self._nearest_k = min(nearest_k, word_neighbors.shape[1])

		self._token_regex = re.compile(r"(?u)\b\w+\b")


	def _euclidean_distance(self, vec1, vec2):
		vec_diff = vec2 - vec1
		return np.sqrt(np.dot(vec_diff, vec_diff))

		

	def make_doc_vector(self, text):
		vec = np.zeros(self._vocab_size)
		for token in self._token_regex.findall(text):
			if token not in self._token2id:
				continue
			word_ind = self._token2id[token]
			vec[word_ind] += 1
			for i in range(self._nearest_k):
				near_word_ind = self._word_neighbors[word_ind][i]
				if near_word_ind == word_ind:
					continue
				vec[near_word_ind] = 1 / self._euclidean_distance(self._word_embeddings[word_ind], self._word_embeddings[near_word_ind])
		return np.log(vec+1)

