
import numpy as np
import sys
import os
import json
import heapq
import pickle
import torch

with open('../largedata/word2vec_vectors.npy', 'rb') as f:
	wv_orig = torch.FloatTensor(np.load(f)).cuda()

print(wv_orig.shape)

num_adj = 1000
adj_matrix = np.zeros((len(wv_orig), num_adj), dtype=np.int32)
adj_matrix_d = np.zeros((len(wv_orig), num_adj), dtype=np.float32)

for i in range(len(wv_orig)):
	sub_arr = wv_orig - wv_orig[i]
	dists2 = torch.sum(sub_arr*sub_arr, dim=1)
	best_wvs = torch.topk(dists2, num_adj, largest=False, sorted=True)

	adj_matrix_d[i][:] = best_wvs[0].float()
	adj_matrix[i][:] = best_wvs[1].int()

	if i % 50 == 0:
		print('\r{}'.format(i), end='')
print()
print('Saving...')

with open('../largedata/word2vec_adjacencies1.npy', 'wb') as f:
	np.save(f, adj_matrix)
with open('../largedata/word2vec_adjacencies_d1.npy', 'wb') as f:
	np.save(f, adj_matrix_d)

print('\nDone.')


