#!/usr/bin/python3

import numpy as np
import json
import cntk as C
import sys
from subprocess import call
from SvmRankFeatureFactory import SvmRankFeatureFactory

class SvmRankTrainer:

	def __init__(self):
		query_docs_file = open('./query_training_set.json', 'r')
		self._feature_factory = SvmRankFeatureFactory()
		self._query_docs = json.load(query_docs_file)
		self._train_data_file = open('./train.dat', 'w')
		self._model_file_name = './model'

	def train(self):
		self.createTrainingDataFile()
		call('svm_rank/svm_rank_learn -c 3 {0} {1}'.format(self._train_data_file.name, self._model_file_name), shell=True)

	def createTrainingDataFile(self):
		train_data = ""
		total_training_instances = 0

		qid = 1
		print("{0} queries".format(len(self._query_docs.keys())))
		
		for query, doc_ids in self._query_docs.items():
			print("Query {0}: {1}".format(qid, query))
			print("doc_ids: {0}".format(doc_ids))
			target = len(doc_ids)
			for doc_id in doc_ids:
				feature_vec = self._feature_factory.getFeatureVec(query, doc_id)
				if feature_vec is not None:
					train_data += self.createSingleTrainingInstance(target, qid, feature_vec)
					total_training_instances += 1
				target -= 1
			qid += 1	

		self._train_data_file.write(train_data)	
		print("{0} queries processed".format(qid-1))
		print("{0} training instances created".format(total_training_instances))
	
	def createSingleTrainingInstance(self, target, qid, feature_vec):
		feature_vec_str = ""
		i = 1
		for _, v in np.ndenumerate(feature_vec):
			feature_vec_str += " " + str(i) + ":" + str(v)
			i += 1

		training_instance  = "{0} qid:{1} {2}\n".format(target, qid, feature_vec_str)
		return training_instance

if __name__ == "__main__":
	trainer = SvmRankTrainer()
	trainer.train()
	

