
import cntk as C
import numpy as np
import json
import pickle
import sys
import os

from argparse import ArgumentParser
from TextTrainingData import TextTrainingData
from DirectEmbedderMinibatchSource import DirectEmbedderMinibatchSource
from cntk.train import Trainer
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.train.training_session import CheckpointConfig, training_session, minibatch_size_schedule


hidden_dim = 250
learning_rate = 0.0025
clipping_threshold_per_sample = 5.0
num_epochs = 10

default_training_data_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata', 'text_training_data.pickle')
default_model_save_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata', 'directdoc2vec_checkpoint.dnn')

parser = ArgumentParser()
parser.add_argument('--training_data_file', dest='training_data_file', action='store', default=default_training_data_filename)
parser.add_argument('--model_save_file', dest='model_save_file', action='store', default=default_model_save_filename)
parser.add_argument('--rcnn_filter_width', dest='rcnn_filter_width', type=int, action='store', default=3)
parser.add_argument('--minibatch_size', dest='minibatch_size', type=int, action='store', default=128)
parser.add_argument('--train_word_embeddings', dest='train_word_embeddings', action='store_true', default=False)
parser.add_argument('--doc_vector_init_file', dest='doc_vector_init_file', action='store', default='')
cmdargs = parser.parse_args(sys.argv[1:])


## Note the order of inputs to match the order of streams in
# ParagraphMinibatchSource: [Label, DocId, ContextWords...]
def create_inputs(hidden_dim, num_docs):
	inputs = []
	label_vector = C.ops.input_variable(hidden_dim, np.float32, is_sparse=False, name='label_input')
	input_text_vector = C.sequence.input_variable((cmdargs.rcnn_filter_width,), dtype=np.float32, name='text_input')
	return [label_vector, input_text_vector]

def create_model(input_list, freq_list, hidden_dim):

	# Embed using recurrent character-level convolutions over the input

	# Placeholder for the input char codes
	window_input = C.ops.placeholder(cmdargs.rcnn_filter_width)

	# Reuse weights when embedding each char
	char_embedding_type = C.layers.Embedding(48)

	# Split up the list of char codes into individual characters so
	# we can embed each one
	char_embeds_list = []
	for i in range(cmdargs.rcnn_filter_width):
		char_input = C.ops.slice(window_input, 0, i, i+1)
		onehot_embedding = C.ops.one_hot(char_input, 128, sparse_output=True)
		char_embeds_list.append(char_embedding_type(onehot_embedding))
	# Put the individual characters back together
	char_embeds = C.ops.splice(*char_embeds_list)

	# Do some initial Dense layers to process each frame
	# This is called sliding_conv because each one is called on
	# just rcnn_filter_width characters for each element of the
	# sequence, as the filter "slides" (via LSTM) across the whole
	# input string
	sliding_convs = C.layers.Sequential([
		C.layers.Dense(32, activation=C.ops.tanh),
		C.layers.Dense(64, activation=C.ops.tanh),
	])(char_embeds)

	# Collect the Dense outputs for the sequence together into a single vector
	lstm1 = C.layers.Recurrence(C.layers.LSTM(64))(sliding_convs)
	lstm2 = C.layers.Fold(C.layers.LSTM(256))(lstm1)
	dense3 = C.layers.Dense(256)(lstm2)
	dense4 = C.layers.Dense(512, activation=C.ops.relu)(dense3)

	# Finally, use a Dense layer to adjust the output size to be
	# hidden_dim and use this as the word embedding
	doc_embedding_type = C.layers.Dense(hidden_dim, activation=None, name="doc_embed")(dense3)

	text_embedding = word_embedding_type(input_list[1])

	mse = C.sum(C.square(text_embedding - input_list[0]))
	rmse = C.sqrt(mse)

	return (text_embedding, mse, rmse)

def train():
	print('Unpickling training data (this could take a short while)')
	with open(cmdargs.training_data_file, 'rb') as f:
		training_data = pickle.load(f)
	with open(cmdargs.doc_vector_init_file, 'rb') as f:
		doc_vectors = np.load(f)

	print('Done unpickling. Final # of training words: {}'.format(training_data.total_words()))

	freq_list = training_data.id2freq
	token2id = training_data.token2id
	#vocab_dim = len(freq_list)
	num_docs = len(training_data.docs)
	print('Training direct doc vectors for {:d} documents'.format(num_docs))

	input_list = create_inputs(hidden_dim, num_docs)

	mb_source = DirectEmbedderMinibatchSource(training_data, {v:k for k,v in token2id.items()}, input_list, cmdargs.rcnn_filter_width, doc_vectors)

	mb_num_samples = cmdargs.minibatch_size
	mb_size = minibatch_size_schedule(mb_num_samples)
	epoch_size = training_data.total_words()//2

	z, cross_entropy, error = create_model(input_list, freq_list, hidden_dim) 

#	lr_schedule = learning_rate_schedule([(3e-3)*(0.8**i) for i in range(12)], UnitType.sample, epoch_size = epoch_size)
#	gradient_clipping_with_truncation = True
#	learner = C.learners.sgd(z.parameters, lr=lr_schedule,
#			    gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
#			    gradient_clipping_with_truncation=gradient_clipping_with_truncation)
	learner = C.learners.adadelta(z.parameters)

	progress_printer = C.logging.ProgressPrinter(freq=20, tag='Training')
	checkpoint_config = CheckpointConfig(frequency = 5*mb_num_samples,
                                           filename = cmdargs.model_save_file,
                                           restore = False)

	trainer = Trainer(z, (cross_entropy, error), [learner], progress_writers=[progress_printer])
	
	input_streams = mb_source.stream_infos()
	input_map = {}
	for i in range(len(input_list)):
		input_map[input_list[i]] = input_streams[i]

	session = training_session(trainer, mb_source, mb_size, input_map, progress_frequency=training_data.total_words(), max_samples = None, checkpoint_config=checkpoint_config, cv_config=None, test_config=None)
	
	C.logging.log_number_of_parameters(z) ; print()
	session.train()


if __name__ == '__main__':
	train()
