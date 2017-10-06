import cntk as C
import numpy as np
import pickle
import os
import sys

from argparse import ArgumentParser
from SampledSoftmax import cross_entropy_with_sampled_softmax
from TextTrainingData import TextTrainingData
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


parser = ArgumentParser()
parser.add_argument('--word_embedding_file', dest='word_embedding_file', action='store', default='')
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
	input_word_vectors = [C.ops.input_variable(vocab_dim, np.float32, is_sparse=True) for i in range(context_size)]
	return [label_vector, docid_vector] + input_word_vectors

def create_model(input_list, freq_list, vocab_dim, hidden_dim):

	# Word embedding layers init
	word_embedding_type = None
	if cmdargs.word_embedding_file == '':
		word_embedding_type = C.layers.Embedding(hidden_dim)
	else:
		word_embeddings = None
		with open(cmdargs.word_embedding_file, 'rb') as f:
			word_embeddings = np.load(f)
		if cmdargs.train_word_embeddings:
			word_embedding_type = C.layers.Embedding(hidden_dim, init=word_embeddings)
		else:
			print(word_embeddings.shape)
			word_embedding_type = C.layers.Embedding(weights=word_embeddings)

	# Document embedding layer init
	doc_embedding = None
	if cmdargs.doc_embedding_init_file == '':
		doc_embedding = C.layers.Embedding(hidden_dim)(input_list[1])
	else:
		doc_embeddings_init = None
		with open(cmdargs.doc_embedding_init_file, 'rb') as f:
			doc_embeddings_init = np.load(f)
		doc_embedding = C.layers.Embedding(hidden_dim, init=doc_embeddings_init)(input_list[1])

	doc_embedding = C.layers.Embedding(hidden_dim)(input_list[1])


	word_embeddings = []
	for i in range(context_size):
		word_embeddings.append(word_embedding_type(input_list[i+2]))

	all_embeddings = [doc_embedding] + word_embeddings
	middle_layer = C.ops.splice(*all_embeddings, axis=0)

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


def do_subsampling(text_training_data, subsampling=1e-5, prog_freq=1e8):
	total_freq = sum(text_training_data.id2freq)
	normalized_id2freq = np.array(text_training_data.id2freq, dtype=np.float64) / total_freq

	text = text_training_data.docs[0]
	indexes_to_remove = []

	# Use batching to let Numpy vectorize and improve performance
	# This is over 5x faster comparted to without batching
	batch_size = 5000

	for i in range(len(text)//batch_size):
		word_ids = text[i*batch_size:i*batch_size+batch_size]
		nWords = len(word_ids)
		removal_probs = 1 - np.sqrt(subsampling / normalized_id2freq[word_ids])
		indexes_to_remove.extend(np.where(np.random.random(size=nWords) < removal_probs)[0]+(i*batch_size))
		if (i*batch_size) % prog_freq < batch_size:
			print('Processed {} ({:0.3f}%) so far. {} words for removal ({:0.1f}%).'.format(i*batch_size, 100.0*i*batch_size/len(text), len(indexes_to_remove), 100.0*len(indexes_to_remove)/(i*batch_size+1)))

	print('Processing {} word removals ({:0.2f}%)...'.format(len(indexes_to_remove), 100.0*len(indexes_to_remove)/len(text)))
	text_training_data.docs[0] = TextTrainingData.remove_indexes(indexes_to_remove)


def train():
	print('Unpickling data (this could take a short while)')
	training_data = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata', 'text_training_data.pickle'), 'rb'))
	#print('Preprocessing data (this could take a LONG while)...')
	#do_subsampling(training_data, subsampling=subsampling_rate, prog_freq=1e7)
	print('Preprocessing is done. Final # of training words: {}'.format(training_data.total_words()))
	mb_source = ParagraphMinibatchSource(training_data, context_size)
	mb_num_samples = 128
	mb_size = minibatch_size_schedule(mb_num_samples)

	freq_list = training_data.id2freq
	token2id = training_data.token2id
	vocab_dim = len(freq_list)
	num_docs = len(training_data.docs)
	print(vocab_dim)
	input_list = create_inputs(vocab_dim, num_docs)

	z, cross_entropy, error = create_model(input_list, freq_list, vocab_dim, hidden_dim) 

	lr_schedule = learning_rate_schedule(learning_rate, UnitType.sample)
	lr_schedule2 = learning_rate_schedule([(3e-3)*(0.8**i) for i in range(10)], UnitType.sample, epoch_size=training_data.total_words()//2)
	mom_schedule = C.learners.momentum_schedule(0.005, UnitType.sample)
	gradient_clipping_with_truncation = True
	learner = C.learners.sgd(z.parameters, lr=lr_schedule2,
			    gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
			    gradient_clipping_with_truncation=gradient_clipping_with_truncation)

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
                                           filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'largedata', "word2vec_checkpoint"),
                                           restore = False)

	trainer = Trainer(z, (cross_entropy, error), [learner], progress_writers=[progress_printer])
	
	input_streams = mb_source.stream_infos()
	input_map = {}
	for i in range(len(input_list)):
		input_map[input_list[i]] = input_streams[i]

	session = training_session(trainer, mb_source, mb_size, input_map, progress_frequency=training_data.total_words(), max_samples = None, checkpoint_config=checkpoint_config, cv_config=None, test_config=None)
	
	C.logging.log_number_of_parameters(z) ; print()
	session.train()
train()

