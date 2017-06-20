#!/usr/bin/python3

# Usage: python3 run_test_queries.py
# Before using, create a test_queries.txt file with one query per line

import custom_query_engine as cqe
import pymongo
from bson.objectid import ObjectId
import re

mongo_client = pymongo.MongoClient('localhost', 27017)

mongo_db = mongo_client['sharesci']
papers_collection = mongo_db['papers']

test_query_file = open('test_queries.txt')

for line in test_query_file.readlines():
	if line == '':
		continue
	print('Performing query: {}'.format(line))
	doc_scores = cqe.process_query(line, max_results=10)
	print('Results:')
	print('{:2s}  {:100s}  {:15s}  {:7s}        '.format('#', 'Title', 'arXiv id', 'Score'))
	result_num = 1
	for result in doc_scores:
		if len(result[0]) == 24:
			mongo_result = papers_collection.find({'_id': ObjectId(result[0])})[0]
			print('{:2d}. {:100s}  {:15s}  {:0.5f}        '.format(result_num, re.sub('[ ]*\n[ ]*', ' ', mongo_result['title']), mongo_result['arXiv_id'], result[1]))
		else:
			print('{:2d}. {:100s}  {:15s}  {:0.5f}        '.format(result_num, result[0], result[0],  result[1]))
		result_num += 1
	print('\n\n\n')

test_query_file.close()

