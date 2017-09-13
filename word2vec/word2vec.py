import cntk as C
import numpy as np
import pickle
import os

from WordMinibatchSource import WordMinibatchSource
from cntk.train import Trainer
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.train.training_session import CheckpointConfig, training_session, minibatch_size_schedule


hidden_dim = 300
alpha = 0.75
num_of_samples = 15
allow_duplicates = False
learning_rate = 0.0025
clipping_threshold_per_sample = 5.0
num_epochs = 10
max_window_size = 3

def cross_entropy_with_sampled_softmax(
    hidden_vector,          
    label_vector,           
    vocab_dim,              
    hidden_dim,             
    num_samples,            
    sampling_weights,       
    allow_duplicates = False 
    ):

	bias = C.layers.Parameter(shape = (vocab_dim, 1), init = 0)
	weights = C.layers.Parameter(shape = (vocab_dim, hidden_dim), init = C.initializer.glorot_uniform())

	sample_selector_sparse = C.random_sample(sampling_weights, num_samples, allow_duplicates)
	sample_selector = sample_selector_sparse

	inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates)
	log_prior = C.log(inclusion_probs)

	wS = C.times(sample_selector, weights, name='wS')
	zS = C.times_transpose(wS, hidden_vector, name='zS1') + C.times(sample_selector, bias, name='zS2') - C.times_transpose (sample_selector, log_prior, name='zS3')

	# Getting the weight vector for the true label. Dimension hidden_dim
	wT = C.times(label_vector, weights, name='wT')
	zT = C.times_transpose(wT, hidden_vector, name='zT1') + C.times(label_vector, bias, name='zT2') - C.times_transpose(label_vector, log_prior, name='zT3')

	zSReduced = C.reduce_log_sum_exp(zS)

	# Compute the cross entropy that is used for training.
	cross_entropy_on_samples = C.log_add_exp(zT, zSReduced) - zT

	# For applying the model we also output a node providing the input for the full softmax
	z = C.times_transpose(weights, hidden_vector) + bias
	z = C.reshape(z, shape = (vocab_dim))

	zSMax = C.reduce_max(zS)
	error_on_samples = C.less(zT, zSMax)

	return (z, cross_entropy_on_samples, error_on_samples)

def create_inputs(vocab_dim):
	input_vector = C.ops.input_variable(vocab_dim, np.float32, is_sparse=True)
	label_vector = C.ops.input_variable(vocab_dim, np.float32, is_sparse=True)
	return input_vector, label_vector

def create_model(input_vector, label_vector, freq_list, vocab_dim, hidden_dim):

	hidden_vector = C.layers.Embedding(hidden_dim)(input_vector)
	#hidden_vector = C.times(input_vector, weights1) + bias1

	smoothed_weights = np.float32(np.power(freq_list, alpha))
	sampling_weights = C.reshape(C.Constant(smoothed_weights), shape = (1,vocab_dim))

	return cross_entropy_with_sampled_softmax(hidden_vector, label_vector, vocab_dim, hidden_dim, num_of_samples, sampling_weights)

def train():
	#training_data = pickle.load(open('tmp_textdata.pickle', 'rb'))
	training_data = pickle.load(open('/dev/shm/tmp194_arxiv.pickle', 'rb'))
	mb_source = WordMinibatchSource(training_data, max_window_size)
	mb_num_samples = 128
	mb_size = minibatch_size_schedule(mb_num_samples)

	freq_list = training_data.id2freq
	token2id = training_data.token2id
	vocab_dim = len(freq_list)
	print(vocab_dim)
	input_vector, label_vector = create_inputs(vocab_dim)

	z, cross_entropy, error = create_model(input_vector, label_vector, freq_list, vocab_dim, hidden_dim) 

	lr_schedule = learning_rate_schedule(learning_rate, UnitType.sample)
	lr_schedule2 = learning_rate_schedule([(3e-3)*(0.8**i) for i in range(10)], UnitType.sample, epoch_size=len(training_data.text_as_id_list)//2)
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
                                           filename = os.path.join(os.getcwd(), "word2vec_checkpoint"),
                                           restore = False)

	trainer = Trainer(z, (cross_entropy, error), [learner], progress_writers=[progress_printer])
	
	input_map = { input_vector: mb_source.fsi, label_vector: mb_source.lsi }	

	session = training_session(trainer, mb_source, mb_size, input_map, progress_frequency=len(training_data.text_as_id_list), max_samples = None, checkpoint_config=checkpoint_config, cv_config=None, test_config=None)
	
	C.logging.log_number_of_parameters(z) ; print()
	session.train()
train()
