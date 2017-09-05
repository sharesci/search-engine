#!/usr/bin/python3

## @file
#

## 
# This script indexes the metadata of documents (title, authors, and abstract)
# using the bigram indexer. In short, it just queries the MongoDB database to
# get the metadata for each document, then indexes each piece of metadata as a
# separate document, using the bigram indexer.
#

import bigram_indexer as bidx
import pymongo
from bson.objectid import ObjectId
import re
import json

#mongo_client = pymongo.MongoClient('137.148.143.48', 27017)

#mongo_db = mongo_client['sharesci']
#papers_collection = mongo_db['papers']

#mongo_result = papers_collection.find()

docs = {}
with open('/tmp/cran.json') as f:
	docs = json.load(f);

token_dict = {}
fulltext_dict = {}
i = 0
#for res in mongo_result:
for res in docs:
	try:
		if 'T' in res and len(res['T']) >= 3:
			token_dict['title_'+str(res['I'])] = res['T']
		if 'B' in res and 3 <= len(res['B']):
			token_dict['abstract_'+str(res['I'])] = res['B']
		if 'A' in res and 0 < len(res['A']):
			# Just make a space-separated list of authors
			token_dict['authors_'+str(res['I'])] = res['A']#'  '.join([' '.join([str(item) for item in author_dict.values() if not isinstance(item, list)]) for author_dict in res['authors'] if isinstance(author_dict, dict)])
		fulltext_dict[str(res['I'])] = res['W']

		# Index batches of documents at a time to improve performance
#		if ((i+1) % 10000) == 0:
#			bidx.index_terms(token_dict, type("", (), {'new_docs': True, 'get_parent_docs': True}))
#			token_dict = {}
#			print('Processed {} total records'.format(i))
		i += 1
	except Exception as e:
		print('An error occurred processing {}'.format(res['I']))
		continue

bidx.index_terms(fulltext_dict, type("", (), {'new_docs': True, 'get_parent_docs': False}))
bidx.index_terms(token_dict, type("", (), {'new_docs': True, 'get_parent_docs': True}))
print('Processed {} total records'.format(i))

