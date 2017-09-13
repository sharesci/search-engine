import cntk as C
from cntk.ops.functions import load_model
import os
import pickle
import numpy as np

def get_token(token2id, token_id):
	return list(token2id.keys())[list(token2id.values()).index(token_id)]

def get_closest_words(sorted_indices, token2id, i, limit):
	words = []
	for j in range(limit):
		words.append(get_token(token2id, sorted_indices[i][j]))
	
	return words


z = load_model(os.path.join(os.getcwd(), "word2vec_checkpoint"))
weights = z.E.value
vocab_dim = weights.shape[0]

max_test_words = 20#vocab_dim
max_closest_words = 10

training_data = pickle.load(open('tmp_textdata.pickle', 'rb'))
token2id = training_data.token2id
id2token = {v: k for k, v in token2id.items()}

test_word_indices = np.random.choice(vocab_dim, max_test_words, replace=False) #Choose random words from vocab
print([max_test_words, vocab_dim])
distance_matrix = np.empty([max_test_words, vocab_dim])

#Calculate Euclidean distance between word vectors
for i in range(max_test_words):
	test_word = weights[test_word_indices[i]]
	for j in range(vocab_dim):
		distance_matrix[i,j] = np.linalg.norm(test_word - weights[j])

sorted_indices = np.argsort(distance_matrix, axis=1)

#Save the result in file
with open('word2vec_test_results.txt', 'a+') as f:
	for i in range(max_test_words):
		words = get_closest_words(sorted_indices, token2id, i, max_closest_words)
		#f.write("\n{0} --> ".format(get_token(token2id, test_word_indices[i])))
		f.write("\n{0} --> ".format(id2token[test_word_indices[i]]))
		for j in range(max_closest_words):
			f.write("{0}, ".format(words[j]))


