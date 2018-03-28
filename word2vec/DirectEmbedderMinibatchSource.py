import cntk as C
import numpy as np


class DirectEmbedderMinibatchSource(C.io.UserMinibatchSource):
	def __init__(self, text_training_data, id2tok, input_list, filter_size, paragraph_vectors):
		self.text_training_data = text_training_data
		self._paragraph_vectors = paragraph_vectors
		self._filter_size = filter_size

		self._pv_dim = len(self._paragraph_vectors[0])

		self.num_docs = len(self.text_training_data.docs)
		self.vocab_dim = len(self.text_training_data.id2freq)
		self.cur_index = 0
		self.id2tok = id2tok
		self._seq_input_vars = input_list

		# Init the streams used to carry the input and output data
		self._init_streams()

		super().__init__()


	def _init_streams(self):
		self.pv_label_si = C.io.StreamInformation("pv_label", 0, 'dense', np.float32, (self._pv_dim,))
		self.text_si = C.io.StreamInformation("text_in", 1, 'dense', np.float32, (3,256))


	def stream_infos(self):
		return [self.pv_label_si, self.text_si]


	def str_to_inputs(self, text_str):
		full_arr = np.zeros(len(text_str)+self._filter_size-1, dtype=np.float32)
		for i in range(len(text_str)):
			full_arr[i] = ord(text_str[i])

		ret_data = np.zeros((len(text_str)+self._filter_size-1, self._filter_size), dtype=np.float32)
		for i in range(len(text_str)):
			ret_data[i+self._filter_size-1] = full_arr[i:i+self._filter_size]
		return ret_data
		


	def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):
		# Note that in this example we do not yet make use of number_of_workers or
		# worker_rank, which will limit the minibatch source to single GPU / single node
		# scenarios.


		docs = self.text_training_data.docs

		docid = -1
		while docid == -1 or len(docs[docid]) == 0:
			docid = np.random.randint(low=0, high=len(docs))
		doc = docs[docid]

		indices = sorted([np.random.randint(len(doc)) for i in range(2)])
		if indices[1] - indices[0] > 100000:
			indices[1] = indices[0] + 100000
		elif indices[1] - indices[0] > 10000 and np.random.random() < 0.3:
			indices[1] = indices[0] + 10000
		if indices[1] - indices[0] < 100:
			indices[1] = indices[0] + 100

		doc_str = '<s>\n' + ' '.join([self.id2tok[word] for word in doc])[indices[0]:indices[1]] + '\n</s>'
		text_data = [self.str_to_inputs(doc_str)]
		pv_label_data = [self._paragraph_vectors[docid]]

		pv_label_data = C.Value.create(self._seq_input_vars[0], pv_label_data)
		text_data = C.Value.create(self._seq_input_vars[1], text_data)

		result = {
			self.pv_label_si: C.io.MinibatchData(pv_label_data, num_samples, num_samples, False),
			self.text_si: C.io.MinibatchData(text_data, num_samples, num_samples, False)
		}

		return result


