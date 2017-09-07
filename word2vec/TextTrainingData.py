#!/usr/bin/python3

import re
import pickle
import numpy as np


class TextTrainingData:
	def __init__(self):
		self.token2id = {}
		self.id2freq = []
		self.text_as_id_list = []
		self.token_regex = re.compile(r"(?u)\b\w+\b")


	def add_text(self, text):
		token_list = self.token_regex.findall(text)
		for token in token_list:
			token_id = 0
			if not (token in self.token2id):
				token_id = len(self.id2freq)
				self.token2id[token] = token_id
				self.id2freq.append(0)
			else:
				token_id = self.token2id[token]

			self.id2freq[token_id] += 1
			self.text_as_id_list.append(token_id)
