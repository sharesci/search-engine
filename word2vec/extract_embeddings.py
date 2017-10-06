#!/usr/bin/env -p python3

import sys
import pickle
import numpy as np
import json
import cntk as C
from argparse import ArgumentParser

def extract_embeddings(model_file, embedding_type):
	if model_file == '' or embedding_type =='':
		return None

	model = C.load_model(model_file)

	embedding_matrices = model.find_all_with_name('E')

	# Extract from a paragraph vector model
	matrix = None
	if embedding_type == 'doc':
		# Note: Take the first 'E' matrix only because that happens to
		# be how they happen to get built; these should be named during
		# model construction in the future
		matrix = embedding_matrices[0]
	else:
		matrix = embedding_matrices[1]

	# Free up some memory
	model = None

	# Return the embeddings as a numpy array
	return np.array(matrix.value, dtype=np.float32)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument(dest='model_file', action='store', type=str, default='')
	parser.add_argument('--embedding_type', dest='embedding_type', action='store', type=str, default='')
	parser.add_argument('--output_file', dest='output_file', action='store', type=str, default='')
	cmdargs = parser.parse_args(sys.argv[1:])

	embeddings = extract_embeddings(cmdargs.model_file, cmdargs.embedding_type)

	if embeddings is None:
		print('Something went wrong extracting the embeddings.')
	else:
		print('Got embeddings with shape {}'.format(str(embeddings.shape)))

	if cmdargs.output_file == '':
		print('No output file. Exiting as dry run.')
		sys.exit(0)

	with open(cmdargs.output_file, 'wb') as f:
		np.save(f, embeddings)

