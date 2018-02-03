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
# Prepare a training set for paragraph2vec with arXiv data that is stored in
# the 'arxiv' collection in MongoDB (running on localhost):
# 
#     python3 save_TextTrainingData.py --data_source mongo --data_location arxiv --paragraph2vec
# 
# 
# Prepare a training set for word2vec consisting of the Cranfield documents in
# this repository:
#
#     python3 save_TextTrainingData.py --data_source cranfield
# 

import pickle
from TextTrainingData import TextTrainingData
import json
import os
import numpy as np
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--paragraph2vec', dest='paragraph2vec', action='store_true', default=False)
parser.add_argument('--data_source_type', dest='data_source_type', action='store', type=str, choices=['arxiv', 'cranfield', 'mongo'], default='arxiv')
parser.add_argument('--data_location', dest='data_location', action='store', type=str, default='', help='The location of the data source. This is the filesystem path for arxiv or cranfield sources, and the collection name for Mongo')
cmdargs = parser.parse_args(sys.argv[1:])

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata')

default_data_locations = {
	'cranfield': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cranfield_data', 'cran.json'),
	'arxiv': '/mnt/data_partition/sharesci/arxiv/preproc/tmp/',
	'mongo': 'papers'
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
			print('Processed {}'.format(i), end='\r')
		i += 1
elif cmdargs.data_source_type == 'mongo':
	import pymongo
	mongo_client = pymongo.MongoClient('localhost', 27017)
	db = mongo_client['sharesci']

	i = 0
	for doc in db[data_location].find({'body': {'$exists': True}}):
		if cmdargs.paragraph2vec:
			data.add_text(doc['body'], doc_name=doc['title'])
		else:
			data.add_text(doc['body'])

		if i % 1000 == 0:
			print('Processed {}'.format(i), end='\r')
		i += 1

print()
print('Deleting infrequent tokens...')
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
