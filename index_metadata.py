#!/usr/bin/python3

## @file
#

## 
# This script indexes the metadata of documents (title, authors, and abstract)
# using the bigram indexer. In short, it just queries the MongoDB database to
# get the metadata for each document, then indexes each piece of metadata as a
# separate document, using the bigram indexer.
#

import custom_query_engine as cqe
import bigram_indexer as bidx
import pymongo
from bson.objectid import ObjectId
import re

mongo_client = pymongo.MongoClient('137.148.143.48', 27017)

mongo_db = mongo_client['sharesci']
papers_collection = mongo_db['papers']

mongo_result = papers_collection.find()

token_dict = {}
i = 1
for res in mongo_result:
	try:
		if 'title' in res and len(res['title']) >= 3:
			token_dict['title_'+str(res['_id'])] = res['title']
		if 'abstract' in res and 3 <= len(res['abstract']):
			token_dict['abstract_'+str(res['_id'])] = res['abstract']
		if 'authors' in res and 0 < len(res['authors']):
			# Just make a space-separated list of authors
			token_dict['authors_'+str(res['_id'])] = '  '.join([' '.join([str(item) for item in author_dict.values() if not isinstance(item, list)]) for author_dict in res['authors'] if isinstance(author_dict, dict)])

		# Index batches of documents at a time to improve performance
		if (i % 10000) == 0:
			bidx.index_terms(token_dict, type("", (), {'new_docs': True, 'get_parent_docs': True}))
			token_dict = {}
			print('Processed {} total records'.format(i))
		i += 1
	except Exception as e:
		print('An error occurred processing {}'.format(res['_id']))
		continue
