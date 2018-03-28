from SvmRankFeatureFactory import SvmRankFeatureFactory
from subprocess import call
import numpy as np

class SvmRankClassifier:

	def __init__(self):
		self._feature_factory = SvmRankFeatureFactory()
		self._test_file_name = './test.dat'
		self._model_file_name = 'model'
		self._result_file_name = 'predictions'

	def classify(self, query, doc_ids):
		self.createTestFile(query, doc_ids)	
		call('svm_rank/svm_rank_classify {0} {1} {2}'.format(self._test_file_name, self._model_file_name, self._result_file_name), shell=True)
		with open(self._result_file_name, 'r') as f:
			content = f.readlines()
		
		doc_scores = { }
		i = 0
		while i < len(doc_ids):
			doc_id = doc_ids[i]
			score = content[i].strip()
			doc_scores[doc_id] = score
			i += 1

		doc_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
		ranked_doc_ids = [doc_score[0] for doc_score in doc_scores]
		return ranked_doc_ids
	
	def createTestFile(self, query, doc_ids):
		test_data = ""
		qid = 1000000
		target = len(doc_ids)
		
		for doc_id in doc_ids:
			feature_vec = self._feature_factory.getFeatureVec(query, doc_id)
			test_data += self.createSingleTestInstance(target, qid, feature_vec)
			target -= 1

		with open(self._test_file_name, 'w') as f:
			f.write(test_data)	

	def createSingleTestInstance(self, target, qid, feature_vec):
		feature_vec_str = ""
		i = 1
		for _, v in np.ndenumerate(feature_vec):
			feature_vec_str += " " + str(i) + ":" + str(v)
			i += 1

		test_instance  = "{0} qid:{1} {2}\n".format(target, qid, feature_vec_str)
		return test_instance

if __name__ == '__main__':
	'''
	Example Usage
	'''
	svmRank = SvmRankClassifier()
	query = 'Cloud Computing'
	doc_ids = ['1703.00374.preproc', '1702.04024.preproc', '1702.07431.preproc' ]
	ranked_doc_ids = svmRank.classify(query, doc_ids)
	print(ranked_doc_ids)

