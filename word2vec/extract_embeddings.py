#!/usr/bin/env -p python3


## @file
#
# This script extracts matrices of word or document embeddings from a saved
# neural network. The options should be mostly self-explanatory.
#
# EXAMPLES:
#
# To extract word embeddings from a normal paragraph vector model:
#     python3 extract_embeddings.py --model_type lookup --output_file embeddings.npy para2vec_checkpoint.dnn word
# The following also works, because the model type is guessed by default:
#     python3 extract_embeddings.py --output_file embeddings.npy para2vec_checkpoint.dnn word
#
# Or extract document embeddings:
#     python3 extract_embeddings.py --output_file doc_embeddings.npy para2vec_checkpoint.dnn doc
#
# Extracting word embeddings from an RCNN model is trickier, since the word
# embeddings are not stored, but instead generated on the fly. However, if we
# give a mapping from tokens to their corresponding IDs, the script can run the
# network to generate the embeddings:
#     python3 extract_embeddings.py --model_type rcnn --token2id_file token2id.json --output_file embeddings.npy para2vec_rcnn_checkpoint.dnn word
#
#
# Note: If the `--output_file` option is ever omitted, the script just does a
# dry run and exits without saving anything.
#


import numpy as np
import cntk as C
import json
from argparse import ArgumentParser
import sys



def str_to_inputs(text_str):
	full_arr = np.zeros(len(text_str)+3-1, dtype=np.float32)
	for i in range(len(text_str)):
		full_arr[i] = ord(text_str[i])

	return full_arr


def extract_rcnn_embedder(full_model):
	embedder_layers = full_model.find_all_with_name("word_embed")
	if len(embedder_layers) == 0:
		raise Exception("Could not get word embedder from RCNN")

	embedder = embedder_layers[0].outputs[0]
	embedder_model = C.ops.combine(embedder.owner)
	return embedder_model

def gen_rcnn_word_embeddings(rcnn_embedder, token2id, batch_size=1000):
	if token2id is None:
		raise ValueError("Need token2id to extract RCNN embeddings")

	id2tok = {v:k for k,v in token2id.items()}
	embeddings = np.zeros((len(id2tok), 250), dtype=np.float32)
	def embed_batch(tok_id_list):
		tmp_embeds = rcnn_embedder.eval({rcnn_embedder.arguments[0]: [str_to_inputs(id2tok[tok_id]) for tok_id in tok_id_list]})
		for k in range(len(tok_id_list)):
			embeddings[tok_id_list[k]] = tmp_embeds[k]

	all_tok_ids = list(id2tok.keys())
	for idx in np.arange(0, len(all_tok_ids), batch_size):
		print("Generating rcnn embeddings...  ({}/{})".format(idx, len(all_tok_ids)), end='\r', file=sys.stderr)
		cur_slice = all_tok_ids[idx:idx+batch_size]
		if len(cur_slice) == 0:
			break
		embed_batch(cur_slice)
	print("Generating rcnn embeddings... ({}/{})".format(len(all_tok_ids), len(all_tok_ids)), file=sys.stderr)
	return embeddings
	

def extract_lookuptable_embeddings(model, embedding_type):

	embedding_matrices = model.find_all_with_name('E')

	# Extract from a paragraph vector model
	matrix = None
	if embedding_type == 'doc':
		# Note: Take the first 'E' matrix only because that happens to
		# be how they happen to get built; these should be named during
		# model construction in the future
		matrix = embedding_matrices[0]
	elif embedding_type == 'word':
		matrix = embedding_matrices[1]
	else:
		raise ValueError("No such embedding type: {}".format(embedding_type))

	# Free up some memory
	model = None

	# Return the embeddings as a numpy array
	return np.array(matrix.value, dtype=np.float32)


def guess_model_type(model):
	if len(model.find_all_with_name("word_embed")) != 0:
		return "rcnn"
	else:
		return "lookup"

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument(dest='model_file', action='store', type=str, default='', help="The file containing the CNTK model")
	parser.add_argument(dest='embedding_type', action='store', type=str, default='', choices=['word', 'doc'], help="The type of embedding to extract")
	parser.add_argument('--model_type', dest='model_type', action='store', type=str, choices=['guess', 'rcnn', 'lookup'], default='guess')
	parser.add_argument('--token2id_file', dest='token2id_file', action='store', type=str, default='../largedata/token2id.json', help="JSON file mapping tokens to IDs")
	parser.add_argument('--output_file', dest='output_file', action='store', type=str, default='')
	cmdargs = parser.parse_args(sys.argv[1:])

	if cmdargs.model_file == '' or cmdargs.embedding_type =='':
		print("You must specify a model file and embedding type", file=sys.stderr)
		sys.exit(1)

	dry_run = False
	if cmdargs.output_file == '':
		print('No output file, so this will just be a dry run. No embeddings will be saved to disk.', file=sys.stderr)
		dry_run = True

	model = C.load_model(cmdargs.model_file)

	model_type = cmdargs.model_type
	if model_type == "guess":
		model_type = guess_model_type(model)

	embeddings = None
	if model_type == "lookup" or cmdargs.embedding_type == "doc":
		embeddings = extract_lookuptable_embeddings(model, cmdargs.embedding_type)
	elif model_type == "rcnn" and cmdargs.embedding_type == "word":
		with open(cmdargs.token2id_file, 'r') as f:
			token2id = json.load(f)
		rcnn_embedder = extract_rcnn_embedder(model)
		embeddings = gen_rcnn_word_embeddings(rcnn_embedder, token2id)
	else:
		print("No method found for extracting embeddings of type {} from a model of type {}".format(cmdargs.embedding_type, model_type), file=sys.stderr)
		sys.exit(1)


	if embeddings is None:
		print('Something went wrong extracting the embeddings.')
	else:
		print('Got embeddings with shape {}'.format(str(embeddings.shape)))

	if dry_run:
		print('Dry run finished. Exiting.')
		sys.exit(0)

	with open(cmdargs.output_file, 'wb') as f:
		np.save(f, embeddings)

