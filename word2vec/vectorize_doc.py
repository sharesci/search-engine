#!/usr/bin/python3

import numpy as np
import pickle
import os
import re
import json
import sys
import scipy.sparse
from QueryEngineCore import ComparatorQueryEngineCore
from DocVectorizer import Word2vecDocVectorizer, TfIdfDocVectorizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--vectorizer_type', dest='vectorizer_type', action='store', type=str, default='word2vec')
parser.add_argument('--no-tsv-output', dest='tsv_output', action='store_false',  default=True)
cmdargs = parser.parse_args(sys.argv[1:])

datasource = 'cranfield'

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata');

with open(os.path.join(data_dir, 'token2id.json'), 'r') as f:
	token2id = json.load(f);

with open(os.path.join(data_dir, 'id2freq.npy'), 'rb') as f:
	id2freq = np.load(f);

with open(os.path.join(data_dir, 'doc2id.json'), 'r') as f:
	doc2id = json.load(f);


if __name__ == '__main__' and datasource == 'cranfield':
	with open('../cranfield_data/cran.json', 'r') as f:
		cran_data = json.load(f)
	with open('../cranfield_data/cran.qrel_full.json', 'r') as f:
		cran_qrel = json.load(f)

	doc_embeds = None
	query_engine = None
	query_vectors = {}
	if cmdargs.vectorizer_type in ['word2vec', 'tfidf']:
		docs = []
		for doc in cran_data:
			docs.append(doc['W'])

		vectorizer = TfIdfDocVectorizer(token2id, len(id2freq))
		if cmdargs.vectorizer_type == 'word2vec':
			vectorizer = Word2vecDocVectorizer(token2id, os.path.join(data_dir, 'word2vec_vectors.npy'))

		doc_embeds = vectorizer.make_doc_embedding_storage(docs)
		query_engine = ComparatorQueryEngineCore(doc_embeds, comparator_func=np.dot)
		for query in cran_qrel:
			query_vec = vectorizer.make_doc_vector(query);
			query_unitvec = query_vec/np.linalg.norm(query_vec)
			query_vectors[query] = query_unitvec
	elif cmdargs.vectorizer_type == 'paragraph2vec':
		para2vec = None
		with open(os.path.join(data_dir, 'paragraph_vectors.npy'), 'rb') as f:
			para2vec = np.load(f)
		docs = []
		for doc in cran_data:
			docs.append(para2vec[doc2id['cran_doc_{}'.format(doc['I'])]])
		doc_embeds = NumpyEmbeddingStorage(np.array(docs))
		def comparator(vec1, vec2):
			sub = np.subtract(vec1,vec2)
			return 15/np.dot(sub, sub)
			#return float(np.dot(vec1, vec2)/np.linalg.norm(vec2))
		query_engine = ComparatorQueryEngineCore(doc_embeds, comparator_func = comparator)
		for query in cran_qrel:
			if len(cran_qrel[query]) == 0:
				query_vectors[query] = np.zeros(300)
				continue
			query_num = int(cran_qrel[query][0]['qnum'])
			query_vectors[query] = para2vec[doc2id['cran_qry_{}'.format(query_num)]]
		#	print(query, query_num)

	nr = dict()
	for query in cran_qrel:
		query_id = cran_qrel[query][0]['qnum']
		nr[query_id] = []

		doc_rels = {}
		for d in cran_qrel[query]:
			doc_rels[int(d['dnum'])] = int(d['rnum'])

		query_unitvec = query_vectors[query]

		results = query_engine.search(query_unitvec)

		for result in results:
			doc_id = int(cran_data[result[1]]['I'])
			doc_rel = doc_rels[doc_id] if doc_id in doc_rels else 5
			nr[query_id].append({'doc_id': doc_id, 'doc_rel': doc_rel, 'score': result[0]})
			if cmdargs.tsv_output:
				print('{:d}\t{:d}\t{:d}\t{:0.16f}'.format(query_id, doc_id, doc_rel, result[0]))
	#with open('/dev/shm/tmp351.json', 'w') as f:
	#	json.dump(nr, f, indent=4, sort_keys=True)


