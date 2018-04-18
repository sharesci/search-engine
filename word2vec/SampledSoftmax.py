import cntk as C
import numpy as np

def cross_entropy_with_sampled_softmax(
    hidden_vector,          
    label_vector,           
    vocab_dim,              
    hidden_dim,             
    num_samples,            
    sampling_weights,       
    allow_duplicates = False,
    weights_init = C.initializer.glorot_uniform(),
    bias_init = 0
    ):

	bias = None
	if bias_init is None:
		# If bias_init is given as None, disable bias completely
		bias = C.layers.Constant(np.zeros((vocab_dim, 1)), name="nce_dense_bias")
	else:
		bias = C.layers.Parameter(shape = (vocab_dim, 1), init = bias_init, name="nce_dense_bias")

	weights = C.layers.Parameter(shape = (vocab_dim, hidden_dim), init = weights_init, name="nce_dense_weights")

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
	cross_entropy_on_samples = C.alias(C.log_add_exp(zT, zSReduced) - zT, name="sampled_cross_entropy")

	# For applying the model we also output a node providing the input for the full softmax
	z = C.times_transpose(weights, hidden_vector) + bias
	z = C.reshape(z, shape = (vocab_dim), name="output_dense")

	zSMax = C.reduce_max(zS)
	error_on_samples = C.less(zT, zSMax, name="sampled_error")

	return (z, cross_entropy_on_samples, error_on_samples)

