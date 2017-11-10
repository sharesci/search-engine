#!/usr/bin/python3

## @file
#
# This script preprocesses a data source and pickles it as a TextTrainingData
# class. In addition, the token2id and id2freq from the TextTrainingData are
# saved separately so they can be accessed without loading all the data into
# RAM.
#
# EXAMPLES:
#
# Prepare arXiv data

import pickle
from TextTrainingData import TextTrainingData
import json
import os
import numpy as np
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--paragraph2vec', dest='paragraph2vec', action='store_true', default=False)
parser.add_argument('--data_source_type', dest='data_source_type', action='store', type=str, choices=['arxiv', 'cranfield'], default='arxiv')
parser.add_argument('--data_location', dest='data_location', action='store', type=str, default='')
cmdargs = parser.parse_args(sys.argv[1:])

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata')

default_data_locations = {
	'cranfield': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cranfield_data', 'cran.json'),
	'arxiv': '/mnt/data_partition/sharesci/arxiv/preproc/tmp/'
}
data_location = cmdargs.data_location
if data_location == '':
	data_location = default_data_locations[cmdargs.data_source_type]


data = TextTrainingData(min_word_freq=5)

if cmdargs.data_source_type == 'cranfield':
	with open('../cranfield_data/cran.json', 'r') as f:
		cran_data = json.load(f)

	for doc in cran_data:
		data.add_text(doc['W'])
elif cmdargs.data_source_type == 'arxiv':
	allfiles = []
	for root, dirs, files in os.walk('/mnt/data_partition/sharesci/arxiv/preproc/tmp/'):
		allfiles.extend([root+name for name in files])

	i = 0
	for filename in allfiles:
		with open(filename, 'r') as f:
			if cmdargs.paragraph2vec:
				data.add_text(f.read(), doc_name=os.path.basename(filename))
			else:
				data.add_text(f.read())
		if i % 1000 == 0:
			print(i)
		i += 1

print('Deleting infrequent tokens')
num_positions_deleted, num_tokens_deleted = data.purge_infrequent_tokens()
print('Deleted {:d} tokens and {:d} positions'.format(num_tokens_deleted, num_positions_deleted))
print('Total vocab size in the end is {:d}'.format(len(data.id2freq)))
print('Total text size in the end is {:d}'.format(data.total_words()))


with open(os.path.join(base_dir, 'text_training_data.pickle'), 'wb') as f:
	pickle.dump(data, f)

with open(os.path.join(base_dir, 'id2freq.npy'), 'wb') as f:
	np.save(f, np.asarray(data.id2freq, dtype=np.int64))

with open(os.path.join(base_dir, 'token2id.json'), 'w') as f:
	json.dump(data.token2id, f, indent=4, sort_keys=True)
