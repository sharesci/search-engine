#!/usr/bin/python3

## @file
#
# This script allows visualizing high-dimensional vectors with a 2D projection
#
# The part warranting the most explanation here is the names_file, which
# specifies which items are to be projected and plotted. It should be a JSON
# file with a structure as follows:
#
# {
#     "group1_label": [
#         "name1",
#         "name2", 
#         ...,
#     ],
#     "group2_label": [
#         "name1",
#         "name2", 
#         ...,
#     ],
#     ...,
# }
#
# In essence, the file should contain an object with each key being the label
# for a group of points to be plotted, and the value for each key is an array
# of the names of the items to be plotted.
#
# EXAMPLES
#
# Visualizing word vectors from word2vec (file names may be different for you)
#     python3 this_file.py my_words.json --legend 'upper left' --weights_file ../largedata/word2vec_vectors.npy --name2id_file ../largedata/token2id.json --annotate_points
#
# Visualizing document vectors:
#     python3 this_file.py my_doc_names.json --legend 'upper left' -- weights_file ../largedata/paragraph_vectors.npy --name2id_file ../largedata/doc2id.json
#

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


def plot_graph(names, projected_vectors, name2id, label='', annotate=False):
	points = np.zeros([len(names), 2])
	for idx, name in enumerate(names):
		vector_id = name2id[name]
		points[idx] = projected_vectors[vector_id]

	#Plot the coordinates of test words
	plt.scatter(points[:,0], points[:,1], label=label)

	#Annotate the points
	if annotate:
		i = 0
		for row in points:
			plt.annotate(names[i], xy=row)
			i += 1


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--annotate_points', dest='annotate_points', help='Annotate the data points', action='store_true', default=False)
	parser.add_argument('--legend', dest='legend', help='Where to put the legend (compatible with pyplot syntax, "none" to have no legend)', action='store', default='none')
	parser.add_argument('--name2id_file', dest='name2id_file', help='File where the name2id JSON is located (e.g., doc2id.json, token2id.json, etc)', action='store', default='../largedata/doc2id.json')
	parser.add_argument('--weights_file', dest='weights_file', help='File where the vector matrix is stored (in Numpy format)', action='store', default='../largedata/paragraph_vectors.npy')
	parser.add_argument('names_file', help='File from which to get the names to test, in JSON format', action='store')
	cmdargs = parser.parse_args(sys.argv[1:])

	with open(cmdargs.weights_file, 'rb') as f:
		weights = np.load(f)
	projected_vectors = TruncatedSVD(n_components = 2).fit_transform(weights)

	with open(cmdargs.names_file, 'r') as f:
		names_to_plot = json.load(f)

	with open(cmdargs.name2id_file, 'r') as f:
		name2id = json.load(f)

	fig = plt.figure(figsize=(6.5,5.5))

	for k in sorted(names_to_plot):
		plot_graph(names_to_plot[k], projected_vectors, name2id, label=k, annotate=cmdargs.annotate_points)

	if cmdargs.legend != 'none':
		plt.legend(loc=cmdargs.legend)

	fig.tight_layout()
	plt.show()


