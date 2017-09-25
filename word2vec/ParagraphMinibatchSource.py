import cntk as C
import numpy as np

class ParagraphMinibatchSource(C.io.UserMinibatchSource):
	def __init__(self, text_training_data, context_size=2):
		self.text_training_data = text_training_data
		self.context_size = context_size

		self.num_docs = len(self.text_training_data.docs)
		self.vocab_dim = len(self.text_training_data.id2freq)
		self.cur_index = 0

		# Init the streams used to carry the input and output data
		self._init_streams()

		super(ParagraphMinibatchSource, self).__init__()


	def _init_streams(self):
		self.label_si = C.io.StreamInformation("labels", 0, 'sparse', np.float32, (self.vocab_dim,))
		self.doc_si = C.io.StreamInformation("doc_id", 1, 'sparse', np.float32, (self.num_docs,))
		self.word_si = []
		for i in range(self.context_size):
			self.word_si.append(C.io.StreamInformation("word_{:d}".format(i), 2+i, 'sparse', np.float32, (self.vocab_dim,)))


	def stream_infos(self):
		return [self.label_si, self.doc_si] + self.word_si


	def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):
		# Note that in this example we do not yet make use of number_of_workers or
		# worker_rank, which will limit the minibatch source to single GPU / single node
		# scenarios.
		docid_data = []
		label_data = []
		# Use comprehension, NOT [[]] * len()
		word_data = [list() for i in range(len(self.word_si))]

		docs = self.text_training_data.docs

		docid = np.random.randint(low=0, high=len(docs))
		doc = docs[docid]
		docid_data = [docid] * num_samples
		
		offsets = np.random.randint(low=0, high=len(doc)-self.context_size-1, size=num_samples)
		for offset in offsets:
			doc_slice = doc[offset:offset+self.context_size+1]
			for i in range(self.context_size):
				word_data[i].append(doc_slice[i])
			label_data.append(doc_slice[self.context_size])

		label_data = C.Value.one_hot(np.array(label_data, dtype=int), self.vocab_dim)
		docid_data = C.Value.one_hot(np.array(docid_data, dtype=int), self.num_docs)
		for i in range(self.context_size):
			word_data[i] = C.Value.one_hot(np.array(word_data[i], dtype=int), self.vocab_dim)

		result = {
			self.label_si: C.io.MinibatchData(label_data, num_samples, num_samples, False),
			self.doc_si: C.io.MinibatchData(docid_data, num_samples, num_samples, False)
		}
		for i in range(self.context_size):
			result[self.word_si[i]] = C.io.MinibatchData(word_data[i], num_samples, num_samples, False)

		return result

