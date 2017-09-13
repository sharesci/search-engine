#!/usr/bin/python3

import pickle
from TextTrainingData import TextTrainingData
import json
import os

datasource = 'arxiv'


data = TextTrainingData(min_word_freq=6)


if datasource == 'cranfield':
	with open('../cranfield_data/cran.json', 'r') as f:
		cran_data = json.load(f)

	for doc in cran_data:
		data.add_text(doc['W'])
	
elif datasource == 'arxiv':
	allfiles = []
	for root, dirs, files in os.walk('/mnt/data_partition/sharesci/arxiv/preproc/tmp/'):
		allfiles.extend([root+name for name in files])

	i = 0
	for filename in allfiles:
		with open(filename, 'r') as f:
			data.add_text(f.read())
		if i % 1000 == 0:
			print(i)
		i += 1

	print('Deleting infrequent tokens')
	num_positions_deleted, num_tokens_deleted = data.purge_infrequent_tokens()
	print('Deleted {:d} tokens and {:d} positions'.format(num_tokens_deleted, num_positions_deleted))
	print('Total vocab size in the end is {:d}'.format(len(data.id2freq)))
	print('Total text size in the end is {:d}'.format(len(data.text_as_id_list)))


with open('tmp_textdata.pickle', 'wb') as f:
	pickle.dump(data, f)
