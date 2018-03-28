import numpy as np
import cntk as C
import json

class SvmRankFeatureFactory:

	def __init__(self):
		self._p_vecs = np.load('../largedata/paragraph_vectors.npy')
		self._doc2id = json.load(open('../largedata/doc2id.json'))
		self._para_vec_model = "/dev/shm/directdoc2vec_checkpoint.dnn"

	def getFeatureVec(self, query, doc_id, doc_type='arxiv'):
		feature_vec = None
		query_vec = self.getQueryVec(query)
		doc_vec = self.getDocumentVec(doc_id, doc_type)
		if doc_vec is not None:
			feature_vec = np.subtract(doc_vec, query_vec)
		return feature_vec

	def getQueryVec(self, query):
		query = "<s>\n" + query + "\n</s>"
		inputs = [self._str_to_inputs(query, 3)]
		z = C.ops.combine(C.load_model(self._para_vec_model).outputs[0])
		input_var = z.arguments[0]
		query_vec = z.eval({input_var: inputs})[0]
		return query_vec

	def getDocumentVec(self, doc_id, doc_type):
		doc_vec = None
		if doc_type == 'arxiv':
			arxiv_id = doc_id
			if not arxiv_id.endswith(".preproc"):
				arxiv_id += ".preproc"

			if arxiv_id in self._doc2id:
				_id = self._doc2id[arxiv_id]
				doc_vec = self._p_vecs[_id]
			else:
				print("Could not find document vector for {0}".format(arxiv_id))
		return doc_vec
	
	def _str_to_inputs(self, text_str, filter_size):
		full_arr = np.zeros(len(text_str)+filter_size-1, dtype=np.float32)
		for i in range(len(text_str)):
			full_arr[i] = ord(text_str[i])

		ret_data = np.zeros((len(text_str)+filter_size-1, filter_size), dtype=np.float32)
		for i in range(len(text_str)):
			ret_data[i+filter_size-1] = full_arr[i:i+filter_size]
		return ret_data

