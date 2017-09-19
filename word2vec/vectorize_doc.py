import numpy as np
import pickle
import os
import re
import json

datasource = 'cranfield'

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');
token_regex = re.compile(r"(?u)\b\w+\b")

with open(os.path.join(data_dir, 'word2vec_vectors.npy'), 'rb') as f:
	word2vec = np.load(f);

with open(os.path.join(data_dir, 'token2id.pickle'), 'rb') as f:
	token2id = pickle.load(f);

def make_doc_vector(text):
	vec = np.zeros(word2vec.shape[1], dtype=np.float64)
	token_list = token_regex.findall(text);
	for token in token_list:
		if token in token2id:
			vec += word2vec[token2id[token]]
	return vec
	
docs = []

if datasource == 'cranfield':
	with open('../cranfield_data/cran.json', 'r') as f:
		cran_data = json.load(f)
	with open('../cranfield_data/cran.qrel_full.json', 'r') as f:
		cran_qrel = json.load(f)

	for doc in cran_data:
		docs.append(make_doc_vector(doc['W']))
		doc['vec'] = docs[-1]

	docs = np.array(docs);
	docs_norm = np.linalg.norm(docs, axis=1).reshape(docs.shape[0], 1)
	docs_norm = np.where(docs_norm==0, 1, docs_norm)
	unitvec_docs = docs/docs_norm

	for query in cran_qrel:
		doc_rels = {}
		for d in cran_qrel[query]:
			doc_rels[int(d['dnum'])] = int(d['rnum'])
		query_vec = make_doc_vector(query);
		query_unitvec = query_vec/np.linalg.norm(query_vec)

		similarities = np.dot(unitvec_docs, query_unitvec)
		for i in range(len(similarities)):
			doc_id = int(cran_data[i]['I'])
			doc_rel = doc_rels[doc_id] if doc_id in doc_rels else 5
			print('{:d}\t{:d}\t{:d}\t{:0.16f}'.format(cran_qrel[query][0]['qnum'], doc_id, doc_rel, similarities[i]))

		#most_similar = cran_data[np.argmax(similarities)]

		#print(query, most_similar['T'], np.max(similarities), end='\n\n+')

