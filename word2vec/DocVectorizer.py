
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
		text = text.lower()

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
	def __init__(self, token2id, vocab_size, word_embeddings, word_neighbors, nearest_k=3):
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
		text = text.lower()

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


class ParagraphVectorDocVectorizer(DocVectorizer):
	def __init__(self, network_file_path, token2id):
		super().__init__();

		self._model, self._cross_entropy, self._error, self._input_list = self._load_saved_model(network_file_path)
		self._context_size = len(self._input_list) - 2
		self._label_input = self._input_list[0]
		self._doc_input = self._input_list[1]

		self._token2id = token2id
		self._vocab_dim = len(self._token2id.keys())

		lr_schedule = C.learners.learning_rate_schedule(2e-3, C.learners.UnitType.sample)
		self._learner = C.learners.sgd(self._model.parameters, lr=lr_schedule,
				    gradient_clipping_threshold_per_sample=5.0,
				    gradient_clipping_with_truncation=True)

		self._trainer = C.train.Trainer(self._model, (self._cross_entropy, self._error), [self._learner], progress_writers=None)

		# Pick an ID to assign for words not in the vocabulary
		# Note: This does overshadow an actual vocab word (unless it
		# was picked as an unknown word placeholder during training),
		# so I picked the _last_ vocab word since it is the most likely
		# to be an uncommon junk word (since it was discovered last)
		self._unknown_word_index = self._vocab_dim - 1

		self._word_regex = re.compile(r"[ ]*(\w+|\.)(?=[ ]|$)")


	def _str_to_inputs(self, text_str):
		full_arr = np.zeros(len(text_str), dtype=np.float32)
		for i in range(len(text_str)):
			full_arr[i] = ord(text_str[i])

		return full_arr


	def make_doc_vectors(self, all_texts):
		all_texts = [text.lower() for text in all_texts]

		# Depending on how big the embedding layer is, we may need to do multiple batches
		num_batches = int(np.ceil(len(all_texts) / self._doc_input.shape[0]))

		final_array = None

		for batch_num in range(num_batches):
			start_index = batch_num * self._doc_input.shape[0]
			end_index = (batch_num + 1) * self._doc_input.shape[0]
			texts = all_texts[start_index:end_index]

			self._reset_model_doc_embeddings(self._model)
			all_words = [self._word_regex.findall(text) for text in texts]
			num_iterations = np.clip(sum(len(words) for words in all_words)//len(all_words), 400, 20000)

			# Save some memory, since we don't need the raw text anymore
			texts = None

			for _ in range(num_iterations):
				# Make a minibatch and train

				input_dict = {input_var: [] for input_var in self._input_list}

				# Randomly sample a context from each doc
				for j in range(len(all_words)):
					if len(all_words[j]) < self._context_size + 2:
						continue

					input_dict[self._doc_input].append(j)
					offset = np.random.randint(len(all_words[j]) - self._context_size - 1)

					input_words = all_words[j][offset:offset+self._context_size]
					label_word = all_words[j][offset+self._context_size]

					for k in range(self._context_size):
						input_dict[self._input_list[2+k]].append(self._str_to_inputs(input_words[k]))

					# For the label, check if the word is known or not
					if label_word in self._token2id:
						input_dict[self._label_input].append(self._token2id[label_word])
					else:
						input_dict[self._label_input].append(self._unknown_word_index)

				input_dict[self._doc_input] = C.Value.one_hot(np.array(input_dict[self._doc_input], dtype=int), self._doc_input.shape[0])
				input_dict[self._label_input] = C.Value.one_hot(np.array(input_dict[self._label_input], dtype=int), self._vocab_dim)

				# Do an update
				self._trainer.train_minibatch(input_dict)

			# Extract the learned value for the doc vector
			cur_array = self._model.find_by_name("doc_embed").E.asarray()[:len(all_words)]
			if final_array is None:
				final_array = cur_array
			else:
				final_array = np.vstack([final_array, cur_array])

		return final_array


	def make_doc_vector(self, text):
		embed_matrix = self.make_doc_vectors([text])
		return embed_matrix[0]


	def make_doc_embedding_storage(self, texts):
		text_embeddings = []
		i = 0
		text_embeddings_arr = np.zeros((len(texts), self._embed_dim))
		for text in texts:
			i += 1
			text_embeddings_arr[i-1][:] = self.make_doc_vector(text)

		return NumpyEmbeddingStorage(text_embeddings_arr)


	## Re-initializes the doc vector being trained by the model
	#
	def _reset_model_doc_embeddings(self, model):
		doc_embed_node = model.find_by_name("doc_embed")
		hidden_dim = doc_embed_node.E.shape[1]

		glorot_uniform_vec = np.random.uniform(-12/hidden_dim, 12/hidden_dim, size=doc_embed_node.E.shape).astype(np.float32)
		doc_embed_node.E.set_value(C.NDArrayView.from_dense(glorot_uniform_vec))

		return model


	## Given a model, extract a list of inputs in the proper order (same order as
	# create_inputs)
	#
	def _get_model_input_list(self, model):
		misc_inputs = [model.find_by_name('label_input'), model.find_by_name("doc_input")]

		num_word_inputs = len(model.arguments) - len(misc_inputs)
		word_inputs = [model.find_by_name("word_input_{}".format(i)) for i in range(num_word_inputs)]

		return misc_inputs + word_inputs 


	def _resize_embedding_layer(self, model, num_embeddings=1):
		doc_embed_node = model.find_by_name("doc_embed")
		hidden_dim = doc_embed_node.output.shape[0]
		doc_input_var = C.input_variable(num_embeddings, is_sparse=True, name="doc_input")
		doc_embed_layer = C.layers.Embedding(hidden_dim, init=C.glorot_uniform(), name="doc_embed")(doc_input_var)

		model = model.clone(C.CloneMethod.freeze, {doc_embed_node: doc_embed_layer})

		cross_entropy = model.find_by_name('sampled_cross_entropy')
		sampled_error = model.find_by_name('sampled_error')

		input_list = self._get_model_input_list(model)

		return model, cross_entropy, sampled_error, input_list


	def _load_saved_model(self, model_filename):
		model = C.load_model(model_filename)

		return self._resize_embedding_layer(model, num_embeddings=1000)



