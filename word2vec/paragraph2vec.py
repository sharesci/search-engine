#!/usr/bin/env -p python3

## @file
#
# Trains a Paragraph Vector model.
#
# EXAMPLES:
# 
# Train a model using the default `text_training_data.pickle` file (which can
# be created using save_TextTrainingData.py), using RCNN to generate word
# embeddings, and learning the word embeddings (as opposed to loading
# pre-trained word embeddings and holding them constant):
# 
#     python3 paragraph2vec.py --word_embedding_method rcnn --train_word_embeddings
#


import cntk as C
import numpy as np
import pickle
import os
import sys

from argparse import ArgumentParser
from SampledSoftmax import cross_entropy_with_sampled_softmax
from TextTrainingData import TextTrainingData
from RcnnParagraphMinibatchSource import RcnnParagraphMinibatchSource
from ParagraphMinibatchSource import ParagraphMinibatchSource
from cntk.train import Trainer
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.train.training_session import CheckpointConfig, training_session, minibatch_size_schedule


hidden_dim = 250
alpha = 0.75
num_of_samples = 15
allow_duplicates = False
learning_rate = 0.0025
clipping_threshold_per_sample = 5.0
num_epochs = 10
context_size = 2
subsampling_rate = 4e-5

default_training_data_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata', 'text_training_data.pickle')
default_model_save_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata', 'para2vec_checkpoint.dnn')

parser = ArgumentParser()
parser.add_argument('--training_data_file', dest='training_data_file', action='store', default=default_training_data_filename)
parser.add_argument('--model_save_file', dest='model_save_file', action='store', default=default_model_save_filename)
parser.add_argument('--word_embedding_file', dest='word_embedding_file', action='store', default='')
parser.add_argument('--word_embedding_method', dest='word_embedding_method', action='store', default='lookup')
parser.add_argument('--rcnn_filter_width', dest='rcnn_filter_width', type=int, action='store', default=3)
parser.add_argument('--minibatch_size', dest='minibatch_size', type=int, action='store', default=128)
parser.add_argument('--train_word_embeddings', dest='train_word_embeddings', action='store_true', default=False)
parser.add_argument('--doc_embedding_init_file', dest='doc_embedding_init_file', action='store', default='')
parser.add_argument('--output_weights_init_file', dest='output_weights_init_file', action='store', default='')
parser.add_argument('--output_bias_init_file', dest='output_bias_init_file', action='store', default='')
cmdargs = parser.parse_args(sys.argv[1:])


## Note the order of inputs to match the order of streams in
# ParagraphMinibatchSource: [Label, DocId, ContextWords...]
def create_inputs(vocab_dim, num_docs):
	inputs = []
	label_vector = C.ops.input_variable(vocab_dim, np.float32, is_sparse=True)
	docid_vector = C.ops.input_variable(num_docs, np.float32, is_sparse=True)
	if cmdargs.word_embedding_method == 'rcnn':
		input_word_vectors = [C.sequence.input_variable((1,), dtype=np.float32, sequence_axis=C.Axis.new_unique_dynamic_axis('word_{}'.format(i))) for i in range(context_size)]
	else:
		input_word_vectors = [C.ops.input_variable(vocab_dim, np.float32, is_sparse=True) for i in range(context_size)]
	return [label_vector, docid_vector] + input_word_vectors

def create_model(input_list, freq_list, vocab_dim, hidden_dim):

	# Word embedding layers init
	word_embedding_type = None
	if cmdargs.word_embedding_method == 'rcnn':
		# Embed using recurrent character-level convolutions over the input

		# Placeholder for the input char codes
		char_input = C.ops.placeholder(1)

		# Reuse weights when embedding each char
		char_embedding_type = C.layers.Embedding(48)

		onehot_embedding = C.ops.one_hot(char_input, 256, sparse_output=True)
		char_embedding = char_embedding_type(onehot_embedding)

		make_windowed_chars = C.layers.Sequential([
			tuple(C.layers.Delay(t-(cmdargs.rcnn_filter_width//2)) for t in range(cmdargs.rcnn_filter_width)),
			C.ops.splice
		])
		all_char_embeds = make_windowed_chars(char_embedding)

		# Do some initial Dense layers to process each frame
		# This is called sliding_conv because each one is called on
		# just rcnn_filter_width characters for each element of the
		# sequence, as the filter "slides" (via LSTM) across the whole
		# input string
		sliding_convs = C.layers.Sequential([
			C.layers.Dense(all_char_embeds.shape, activation=C.ops.tanh),
			C.layers.Dense(all_char_embeds.shape, activation=C.ops.tanh),
		])(all_char_embeds)

		# Residual layer
		res1 = C.ops.plus(all_char_embeds, sliding_convs)

		fc1 = C.layers.Dense(32, activation=C.ops.tanh)(res1)

		# Collect the Dense outputs for the sequence together into a single vector
		lstm1 = C.layers.Fold(C.layers.LSTM(96))(fc1)

		# Finally, use a Dense layer to adjust the output size to be
		# hidden_dim and use this as the word embedding
		word_embedding_type = C.layers.Dense(hidden_dim, name="word_embed")(lstm1)
	elif cmdargs.word_embedding_method == 'lookup' and cmdargs.word_embedding_file != '':
		# Embed using a lookup table (CNTK "Embedding" layer)
		word_embeddings = None
		with open(cmdargs.word_embedding_file, 'rb') as f:
			word_embeddings = np.load(f)
		if cmdargs.train_word_embeddings:
			word_embedding_type = C.layers.Embedding(hidden_dim, init=word_embeddings, name="word_embed")
		else:
			word_embedding_type = C.layers.Embedding(weights=word_embeddings, name="word_embed")
	else:
		# Default to the most basic lookup table method
		word_embedding_type = C.layers.Embedding(hidden_dim, name="word_embed")

	# Document embedding layer init
	doc_embedding = None
	if cmdargs.doc_embedding_init_file == '':
		doc_embedding = C.layers.Embedding(hidden_dim, name="doc_embed")(input_list[1])
	else:
		doc_embeddings_init = None
		with open(cmdargs.doc_embedding_init_file, 'rb') as f:
			doc_embeddings_init = np.load(f)
		doc_embedding = C.layers.Embedding(hidden_dim, init=doc_embeddings_init, name="doc_embed")(input_list[1])


	word_embeddings = []
	for i in range(context_size):
		word_embeddings.append(word_embedding_type(input_list[i+2]))

	all_embeddings = [doc_embedding] + word_embeddings
	middle_layer = C.ops.splice(*all_embeddings, axis=0, name="hidden_splice")

	smoothed_weights = np.float32(np.power(freq_list, alpha))
	sampling_weights = C.reshape(C.Constant(smoothed_weights), shape = (1,vocab_dim))

	output_weights_init = C.initializer.glorot_uniform()
	output_bias_init = 0
	if cmdargs.output_weights_init_file != '':
		with open(cmdargs.output_weights_init_file, 'rb') as f:
			output_weights_init = np.load(f)
	if cmdargs.output_bias_init_file != '':
		with open(cmdargs.output_bias_init_file, 'rb') as f:
			output_bias_init = np.load(f)

	return cross_entropy_with_sampled_softmax(middle_layer, input_list[0], vocab_dim, hidden_dim*len(all_embeddings), num_of_samples, sampling_weights, weights_init = output_weights_init, bias_init = output_bias_init)


def train():
	print('Unpickling training data (this could take a short while)')
	with open(cmdargs.training_data_file, 'rb') as f:
		training_data = pickle.load(f)
	print('Done unpickling. Final # of training words: {}'.format(training_data.total_words()))

	freq_list = training_data.id2freq
	token2id = training_data.token2id
	vocab_dim = len(freq_list)
	num_docs = len(training_data.docs)
	print('Training paragraph vectors for {:d} documents'.format(vocab_dim))

	input_list = create_inputs(vocab_dim, num_docs)

	mb_source = None
	if cmdargs.word_embedding_method == 'rcnn':
		mb_source = RcnnParagraphMinibatchSource(training_data, {v:k for k,v in token2id.items()}, input_list, context_size)
	elif cmdargs.word_embedding_method == 'lookup':
		mb_source = ParagraphMinibatchSource(training_data, context_size)
	else:
		print("Invalid word embedding type.", file=sys.stderr)
		sys.exit(1)

	mb_num_samples = cmdargs.minibatch_size
	mb_size = minibatch_size_schedule(mb_num_samples)
	epoch_size = training_data.total_words()//2

	z, cross_entropy, error = create_model(input_list, freq_list, vocab_dim, hidden_dim) 

	lr_schedule = learning_rate_schedule([(3e-3)*(0.8**i) for i in range(12)], UnitType.sample, epoch_size = epoch_size)
	gradient_clipping_with_truncation = True
	learner = C.learners.sgd(z.parameters, lr=lr_schedule,
			    gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
			    gradient_clipping_with_truncation=gradient_clipping_with_truncation)

#	mom_schedule = C.learners.momentum_schedule(0.005, UnitType.sample)
#	var_mom_schedule = C.learners.momentum_schedule(0.999, UnitType.sample)
#	learner2 = C.learners.adam(z.parameters,
#		lr=lr_schedule,
#		momentum=mom_schedule,
#		variance_momentum=var_mom_schedule,
#		epsilon=1.5e-8,
#		gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
#		gradient_clipping_with_truncation=gradient_clipping_with_truncation)

	progress_printer = C.logging.ProgressPrinter(freq=200, tag='Training')
	checkpoint_config = CheckpointConfig(frequency = 100000*mb_num_samples,
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

