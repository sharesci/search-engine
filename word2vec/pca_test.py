import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from cntk.ops.functions import load_model

test_words = ['http', 'html', 'neural', 'algorithm', 'math', 'science']

def get_model_weights():
	z = load_model(os.path.join(os.getcwd(), "word2vec_checkpoint"))
	weights = z.E.value
	return weights

def reduce_dimentions(weights):
	reduced_weights = PCA(n_components = 2).fit_transform(weights)
	return reduced_weights

def plot_graph(words, all_points, token2id):
	points = np.empty([len(words), 2])

	for idx, word in enumerate(words):
		word_id = token2id[word]
		points[idx] = all_points[word_id]

	#Plot the coordinates of test words
	plt.scatter(points[:,0], points[:,1])

	#Annotate the points
	i = 0
	for row in points:
		plt.annotate(words[i], xy=row)
		i += 1

	plt.show()

weights = get_model_weights()
all_points = reduce_dimentions(weights)

training_data = pickle.load(open('tmp_textdata.pickle', 'rb'))

plot_graph(test_words, all_points, training_data.token2id)



