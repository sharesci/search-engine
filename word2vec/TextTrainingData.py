#!/usr/bin/python3

import re
import pickle
import numpy as np


class TextTrainingData:
	def __init__(self, min_word_freq=2):
		self._min_word_freq = min_word_freq
		self.doc2id = {}
		self.token2id = {}
		self.id2freq = []
		self.docs = []
		self.token_regex = re.compile(r"(?u)\b\w+\b")

		# We'll use this to keep track of positions of infrequent
		# tokens, without having to clutter the real token2id table
		# with them
		self._infrequent_tokens = {}


	def add_text(self, text, doc_name='_'):
		if doc_name not in self.doc2id:
			self.doc2id[doc_name] = len(self.docs)
			self.docs.append([])
		doc_id = self.doc2id[doc_name]

		for token in self.token_regex.findall(text):
			self._add_token(token, doc_id)


	def _add_token(self, token, doc_id):
		text_id_list = self.docs[doc_id]
		if token in self.token2id:
			token_id = self.token2id[token]
			self.id2freq[token_id] += 1
			text_id_list.append(token_id)

			# Important to return here; otherwise the token will be
			# added back to the infrequent_tokens list and possibly
			# deleted
			return

		if token not in self._infrequent_tokens:
			self._infrequent_tokens[token] = []

		self._infrequent_tokens[token].append((doc_id, len(text_id_list)))
		text_id_list.append(-1)

		# Once we have enough appearances, make this a real token
		if len(self._infrequent_tokens[token]) == self._min_word_freq:
			token_id = len(self.id2freq)
			self.token2id[token] = token_id
			self.id2freq.append(len(self._infrequent_tokens[token]))

			# Fill in the placeholders with the real id
			for doc_id2, index in self._infrequent_tokens[token]:
				self.docs[doc_id2][index] = token_id

			self._infrequent_tokens.pop(token, None)


	def total_words(self):
		return sum([len(self.docs[i]) for i in range(len(self.docs))])


	## Get rid of all references to infrequent tokens
	# 
	# This includes both removing them from the word list and clearing the
	# temporary infrequent token dict
	#
	def purge_infrequent_tokens(self):
		# Make sure this is a list comprehension and not something like
		# [[]], or all elements will reference the same list
		all_indexes = [list() for i in range(len(self.docs))]
		for k in self._infrequent_tokens:
			for item in self._infrequent_tokens[k]:
				all_indexes[item[0]].append(item[1])

		for i in range(len(all_indexes)):
			self.docs[i] = TextTrainingData.remove_indexes(self.docs[i], all_indexes[i])

		num_infrequent = len(self._infrequent_tokens.keys())
		self._infrequent_tokens = {}

		return sum([len(d) for d in all_indexes]), num_infrequent


	## Removes the elements at the indexes specified in index_list from
	# main_list
	#
	def remove_indexes(main_list, index_list):
		if len(index_list) == 0:
			return main_list[:]

		index_list = sorted(index_list)

		new_list = main_list[:index_list[0]]
		for i in range(len(index_list)):
			if i < len(index_list)-1:
				new_list.extend(main_list[index_list[i]+1:index_list[i+1]])
			else:
				new_list.extend(main_list[index_list[i]+1:])

		return new_list

