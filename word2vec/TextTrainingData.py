#!/usr/bin/python3

import re
import pickle
import numpy as np


class TextTrainingData:
	def __init__(self, min_word_freq=2):
		self._min_word_freq = min_word_freq
		self.token2id = {}
		self.id2freq = []
		self.text_as_id_list = []
		self.token_regex = re.compile(r"(?u)\b\w+\b")

		# We'll use this to keep track of positions of infrequent
		# tokens, without having to clutter the real token2id table
		# with them
		self._infrequent_tokens = {}


	def add_text(self, text):
		token_list = self.token_regex.findall(text)
		for token in token_list:
			self._add_token(token)


	def _add_token(self, token):
		if token in self.token2id:
			token_id = self.token2id[token]
			self.id2freq[token_id] += 1
			self.text_as_id_list.append(token_id)
			return

		if token not in self._infrequent_tokens:
			self._infrequent_tokens[token] = []

		self._infrequent_tokens[token].append(len(self.text_as_id_list))
		self.text_as_id_list.append(-1)

		# Once we have enough appearances, make this a real token
		if len(self._infrequent_tokens[token]) == self._min_word_freq:
			token_id = len(self.id2freq)
			self.token2id[token] = token_id
			self.id2freq.append(len(self._infrequent_tokens[token]))

			# Fill in the placeholders with the real id
			for index in self._infrequent_tokens[token]:
				self.text_as_id_list[index] = token_id

			self._infrequent_tokens.pop(token, None)



	## Get rid of all references to infrequent tokens
	# 
	# This includes both removing them from the word list and clearing the
	# temporary infrequent token dict
	#
	def purge_infrequent_tokens(self):
		all_indexes = []
		for k in self._infrequent_tokens:
			all_indexes.extend(self._infrequent_tokens[k])

		all_indexes = sorted(all_indexes)

		new_text_id_list = []
		for i in range(len(all_indexes)):
			if i < len(all_indexes)-1:
				new_text_id_list.extend(self.text_as_id_list[all_indexes[i]+1:all_indexes[i+1]])
			else:
				new_text_id_list.extend(self.text_as_id_list[all_indexes[i]+1:])
			#del self.text_as_id_list[index]

		self.text_as_id_list = new_text_id_list

		num_infrequent = len(self._infrequent_tokens.keys())
		self._infrequent_tokens = {}

		return len(all_indexes), num_infrequent
