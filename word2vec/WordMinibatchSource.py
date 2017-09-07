import cntk as C
import numpy as np

class WordMinibatchSource(C.io.UserMinibatchSource):
	def __init__(self, text_training_data, max_window_size):
		self.text_training_data = text_training_data
		self.max_window_size = max_window_size

		self.vocab_dim = len(self.text_training_data.id2freq)
		self.cur_index = 0

		# The streams used to carry the actual input and output data
		self.fsi = C.io.StreamInformation("features", 0, 'sparse', np.float32, (self.vocab_dim,))
		self.lsi = C.io.StreamInformation("labels", 1, 'sparse', np.float32, (self.vocab_dim,))

		# To store extra features/labels if we generate too many
		self.leftover_features = []
		self.leftover_labels = []

		super(WordMinibatchSource, self).__init__()

	def stream_infos(self):
		return [self.fsi, self.lsi]

	def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):
		# Note that in this example we do not yet make use of number_of_workers or
		# worker_rank, which will limit the minibatch source to single GPU / single node
		# scenarios.
		f_data = []
		l_data = []
		text = self.text_training_data.text_as_id_list

		f_data.extend(self.leftover_features)
		l_data.extend(self.leftover_labels)

		while len(f_data) < num_samples:
			# Choose the target "context" word first and then iterate over
			# possible input words, because for some reason that's how the
			# original C implementation seems to do it (?)
			target_word = text[self.cur_index]

			window_size = np.random.randint(self.max_window_size)
			span_size = window_size * 2 + 1

			window_start = max(self.cur_index - window_size, 0)
			window_end = window_start + span_size

			new_features = text[window_start:window_end]
			# Use ``len(new_features)`` instead of ``span`` for the size in
			# case the window was cut off by the text boundary
			new_labels = [target_word] * len(new_features)

			f_data.extend(new_features)
			l_data.extend(new_labels)

			self.cur_index += 1
			if self.cur_index >= len(text):
				self.cur_index = 0

		# Store leftover samples for the next minibatch
		self.leftover_features = f_data[num_samples:]
		self.leftover_labels = l_data[num_samples:]

		f_data = C.Value.one_hot(np.array(f_data[:num_samples], dtype=int), self.vocab_dim)
		l_data = C.Value.one_hot(np.array(l_data[:num_samples], dtype=int), self.vocab_dim)

		result = {
			self.fsi: C.io.MinibatchData(f_data, num_samples, num_samples, False),
			self.lsi: C.io.MinibatchData(l_data, num_samples, num_samples, False)
		}
		return result
