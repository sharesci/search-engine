#!/usr/bin/python3

## @file
#


import custom_query_engine as cqe
import pymongo
from bson.objectid import ObjectId
import re
import heapq
import numpy as np
import copy
import json


def init_mappings(mappings_filename):
	mappings = {}
	with open(mappings_filename) as f:
		for m in json.load(f):
			mappings[m['arXiv_id']] = m['_id']
	return mappings


def calc_fitness(results, relevant_results):
	count = 0
	i = 1
	for result in results:
		if result[0] in relevant_results:
			count += 1/np.sqrt(i)
		i += 1
	return count


def evaluate_set(train_set, params):
	fitness = 0.0
	for query in train_set.keys():
		if len(train_set[query]) == 0:
			continue
		results = cqe.process_query(query, max_results=50, weights=params, print_idfs=False)
		fitness += calc_fitness(results, train_set[query])

	return fitness


def generate_child_params(parent_params, max_children, step_size=0.1):
	children = [copy.deepcopy(parent_params)]
	while len(children) < max_children:
		child_params = copy.deepcopy(parent_params)

		# Make random changes
		for param_key in child_params.keys():
			cur_val = parent_params[param_key]
			new_val = cur_val + np.random.uniform(-cur_val*step_size, cur_val*step_size)
			child_params[param_key] = new_val

		# Normalize
		total_val = sum(child_params.values())
		for param_key in child_params.keys():
			child_params[param_key] = child_params[param_key] / total_val

		children.append(child_params)
	return children


def get_k_fittest_candidates(train_set, candidate_params, k):

	fitness = []
	for params_num in range(len(candidate_params)):
		print("\rEvaluating {:d}/{:d}".format(params_num+1, len(candidate_params)), end="")
		fitness.append((evaluate_set(train_set, candidate_params[params_num]), params_num))

	print()

	return heapq.nlargest(k, fitness, key=lambda x: x[0])


def do_evolution_iteration(train_set, candidate_params, step_size=0.1, evolution_max_k=3):

	fittest_k = get_k_fittest_candidates(train_set, candidate_params, evolution_max_k)
	fittest_k_idxs = [tup[1] for tup in fittest_k]
	print('\nFittest: {}\n'.format(str([(tup[0], tup[1], candidate_params[tup[1]]) for tup in fittest_k])))

	child_params = []
	for fit_idx in fittest_k_idxs:
		child_params.extend(generate_child_params(candidate_params[fit_idx], 5, step_size))

	return child_params


def optimize_query_params(train_set, initial_params, step_size=0.15, evolution_max_k=5, num_iterations=40):

	candidate_params = [initial_params]
	for iteration_num in range(num_iterations):
		candidate_params = do_evolution_iteration(train_set, candidate_params, step_size, evolution_max_k)

	return candidate_params[get_k_fittest_candidates(train_set, candidate_params, 1)[0][1]]


if __name__ == '__main__':
	train_set = {}
	with open('query_training_set.json') as f:
		train_set = json.load(f)

#	mappings = init_mappings('results2.json')
#	for query in train_set.keys():
#		new_ids = []
#		for arxiv_id in train_set[query]:
#			if arxiv_id in mappings.keys():
#				new_ids.append(mappings[arxiv_id])
#			else:
#				new_ids.append(arxiv_id)
#		train_set[query] = new_ids
	initial_params = cqe.DEFAULT_QUERY_WEIGHTS

	print(optimize_query_params(train_set, initial_params, step_size=0.2, num_iterations=40, evolution_max_k=5))

