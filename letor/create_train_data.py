import numpy as np
import json
import cntk as C
import sys

def str_to_inputs(text_str, filter_size):
	full_arr = np.zeros(len(text_str)+filter_size-1, dtype=np.float32)
	for i in range(len(text_str)):
		full_arr[i] = ord(text_str[i])

	ret_data = np.zeros((len(text_str)+filter_size-1, filter_size), dtype=np.float32)
	for i in range(len(text_str)):
		ret_data[i+filter_size-1] = full_arr[i:i+filter_size]
	return ret_data

def getFeatureVec(doc_vec, query_vec):
	return np.subtract(doc_vec, query_vec)

def getQueryVec(query):
	query = "<s>\n" + query + "\n</s>"
	inputs = [str_to_inputs(query, 3)]
	z = C.ops.combine(C.load_model("/dev/shm/directdoc2vec_checkpoint.dnn").outputs[0])
	input_var = z.arguments[0]
	query_vec = z.eval({input_var: inputs})[0]
	
	return query_vec

def getDocumentVec(arxiv_id):
	if not arxiv_id.endswith(".preproc"):
		arxiv_id += ".preproc"

	p_vecs = np.load('../largedata/paragraph_vectors.npy')
	doc2id = json.load(open('../largedata/doc2id.json'))
	doc_id = None
	if arxiv_id in doc2id:
		doc_id = doc2id[arxiv_id]
		return p_vecs[doc_id]

	print("Could not find document vector for {0}".format(arxiv_id))
	return None

def createSingleTrainingInstance(target, qid, feature_vec):
	feature_vec_str = ""
	i = 1
	for _, v in np.ndenumerate(feature_vec):
		feature_vec_str += " " + str(i) + ":" + str(v)
		i += 1

	training_instance  = "{0} qid:{1} {2}\n".format(target, qid, feature_vec_str)
	return training_instance

def createTrainingDataFile(query_docs):
	train_data_file = open('./train.dat', 'w+')
	train_data = ""
	total_training_instances = 0

	qid = 1
	print("{0} queries".format(len(query_docs.keys())))
	
	for query, doc_ids in query_docs.items():
		query_vec = getQueryVec(query)
		print("Query {0}: {1}".format(qid, query))
		print("doc_ids: {0}".format(doc_ids))
		target = len(doc_ids)
		for doc_id in doc_ids:
			doc_vec = getDocumentVec(doc_id)
			if doc_vec is not None:
				feature_vec = getFeatureVec(doc_vec, query_vec)
				train_data += createSingleTrainingInstance(target, qid, feature_vec)		
				total_training_instances += 1
			target -= 1
		qid += 1	

	train_data_file.write(train_data)	
	print("{0} queries processed".format(qid-1))
	print("{0} training instances created".format(total_training_instances))


if __name__ == "__main__":
	query_docs_file = open('../query_training_set.json', 'r')
	query_docs = json.load(query_docs_file)
	createTrainingDataFile(query_docs)
	

