import os
import pickle
import numpy as np
import heapq
import sys
import json

# This script gets the nearest neighbors of the word entered at the prompt.
# Run with --random-words to get the old behavior of selecting a group of
# random words instead of prompting.

with open('../largedata/word2vec_vectors.npy', 'rb') as f:
	weights = np.load(f)
vocab_dim = weights.shape[0]

max_test_words = 20
max_closest_words = 20

with open('../largedata/token2id.json', 'r') as f:
	token2id = json.load(f)
id2token = {v: k for k, v in token2id.items()}

# Calculate Euclidean distance between word vectors
def print_closest(wordvec, max_closest=2):
	myheap = [(-sys.maxsize, -sys.maxsize)] * max_closest
	heapq.heapify(myheap)
	for j in range(vocab_dim):
		heapq.heappushpop(myheap, (-np.linalg.norm(wordvec - weights[j]), j))

	closest_words = heapq.nlargest(max_closest, myheap)
	for i in range(len(closest_words)):
		wordtuple = closest_words[i]
		if wordtuple[1] == -1:
			continue
		print('{}'.format(id2token[wordtuple[1]]), end='')
		if i < len(closest_words)-1:
			print(', ', end='')

def interactive_mode():
	qw = ''

	while qw != 'exit':
		try:
			qw = input('Type a word: ');
		except EOFError as err:
			print('exit')
			break

		allwords = qw.split('+')

		if any([(word not in token2id) for word in allwords]):
			print('One of those words is not vectorized.')
			continue

		total_wordvec = np.zeros_like(weights[0], dtype=np.float32)
		for word in allwords:
			word_id = token2id[word]
			total_wordvec += np.array(weights[word_id], dtype=np.float32)

		print('{} ->  '.format(qw), end='')
		print_closest(total_wordvec, max_closest_words)
		print()


def random_mode():
	for i in range(20):
		word_id = np.random.randint(low=0,high=vocab_dim)
		word_vec = weights[word_id]
		print('{} ->  '.format(id2token[word_id]), end='')
		print_closest(word_vec, max_closest_words)
		print()
		


if __name__ == '__main__':
	if '--random-words' in sys.argv:
		random_mode()
	else:
		interactive_mode()

