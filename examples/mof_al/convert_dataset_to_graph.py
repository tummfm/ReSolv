"""Helper file that converts a dataset of particle positions and boxes to the
SparseGraph representation of DimeNet.
"""
import os
import pickle
import time

os.environ['CUDA_VISIBLE_DEVICES'] = str()  # CPU here often faster

import jax.numpy as jnp
import numpy as onp

from chemtrain import data_processing, sparse_graph

save_str = '../../../../Datasets/LJ/padded_graph_LJ_10k.pkl'

configuration_str = '../../../../Datasets/LJ/conf_atoms_LJ_10k.npy'
length_str = '../../../../Datasets/LJ/length_atoms_LJ_10k.npy'
box_side_length = onp.load(length_str)
box = jnp.ones(3) * box_side_length

position_data = data_processing.get_dataset(configuration_str, retain=200)
num_samples, n_particles, _ = position_data.shape
species = onp.zeros(n_particles)

r_cut = 1.5  # 1.5 sigma in LJ units

start = time.time()
graph = sparse_graph.convert_dataset_to_graphs(r_cut, position_data, box,
                                               species)
print(f'Time to process {num_samples} samples:'
      f' {time.time() - start} s')

with open(save_str, 'wb') as f:
    pickle.dump(graph, f)
