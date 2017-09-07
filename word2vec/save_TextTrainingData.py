#!/usr/bin/python3

import pickle
from TextTrainingData import TextTrainingData
import json

with open('../cranfield_data/cran.json', 'r') as f:
	cran_data = json.load(f)

data = TextTrainingData()
for doc in cran_data:
	data.add_text(doc['W'])

with open('tmp_textdata.pickle', 'wb') as f:
	pickle.dump(data, f)
