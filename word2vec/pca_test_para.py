import os
import pickle
import json
import re
import numpy as np
import time
import gc
import sys
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
import scipy.sparse
from argparse import ArgumentParser



def get_model_weights():
	with open('../largedata/paragraph_vectors.npy', 'rb') as f:
		weights = np.load(f)
	return weights


def reduce_dimentions(weights):
	return TruncatedSVD(n_components = 2).fit_transform(weights)


def plot_graph(doc_names, all_points, doc2id, label='', annotate=False):
	points = np.zeros([len(doc_names), 2])
	for idx, doc in enumerate(doc_names):
		doc_id = doc2id[doc]
		points[idx] = all_points[doc_id]

	#Plot the coordinates of test words
	plt.scatter(points[:,0], points[:,1], label=label)

	#Annotate the points
	if annotate:
		i = 0
		for row in points:
			plt.annotate(doc_names[i], xy=row)
			i += 1


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--annotate_points', dest='annotate_points', help='Annotate the data points', action='store_true', default=False)
	parser.add_argument('--legend', dest='legend', help='Where to put the legend (compatible with pyplot syntax, "none" to have no legend)', action='store', default='none')
	parser.add_argument('--doc2id_file', dest='doc2id_file', help='File where the doc2id JSON is located', action='store', default='../largedata/doc2id.json')
	parser.add_argument('names_file', help='File from which to get the doc names to test, in JSON format', action='store')
	cmdargs = parser.parse_args(sys.argv[1:])

	weights = get_model_weights()
	all_points = reduce_dimentions(weights)

	with open(cmdargs.names_file, 'r') as f:
		test_doc_names = json.load(f)

	with open(cmdargs.doc2id_file, 'r') as f:
		doc2id = json.load(f)

	fig = plt.figure(figsize=(6.5,5.5))

	for k in sorted(test_doc_names):
		plot_graph(test_doc_names[k], all_points, doc2id, label=k, annotate=cmdargs.annotate_points)

	if cmdargs.legend != 'none':
		plt.legend(loc=cmdargs.legend)

	fig.tight_layout()
	plt.show()


