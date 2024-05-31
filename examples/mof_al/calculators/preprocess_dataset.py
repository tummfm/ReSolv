"""Example file for preprocessing dataset to padded DimeNet graphs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str()  # CPU here often faster

from jax import config
config.update('jax_disable_jit', True)

import json
from multiprocessing.connection import Client
import numpy as onp
import pickle
import time

from chemtrain import sparse_graph, data_processing


shuffle = True  # shuffle dataset after preprocessing


def load_dataset():
    """Loads the whole dataset from the data-server.

    Requires running mof-q-dataloader.py in background as data-server.
    """
    if 'c' not in locals():
        c = Client('./mofq-socket', authkey=b'test')
        c.send('{ "cmd" : "get_data", "data_id" : "*",'  # ["ZUXPOZ_clean"],'
               '"properties" : ["id", "struc_numbers", "struc_positions",'
               ' "struc_cell", "partial_charges"] }')
        dataset = json.loads(c.recv())
    else:
        raise RuntimeError('Could not open Client.')
    return dataset


save_path = 'partial_charge_dataset.pkl'
r_cut = 5.  # 5 A for dimenet

json_data = load_dataset()

# convert to list of arrays
atom_types = [onp.array(types) for types in
              json_data['properties']['struc_numbers']]
positions = [onp.array(pos) for pos in
             json_data['properties']['struc_positions']]
boxes = onp.array(json_data['properties']['struc_cell'])
partial_charges = [onp.array(charges) for charges in
                   json_data['properties']['partial_charges']]

padded_charges, _ = sparse_graph.pad_per_atom_quantities(partial_charges)


t_start = time.perf_counter()
data_graph = sparse_graph.convert_dataset_to_graphs(r_cut, positions, boxes,
                                                    atom_types)
print(f'Time for processing dataset {time.perf_counter() - t_start} s.')

if shuffle:
    data_graph, _, _ = data_processing.train_val_test_split(
        data_graph, 1., 0., shuffle=True)
    padded_charges, _, _ = data_processing.train_val_test_split(
        padded_charges, 1., 0., shuffle=True)

with open(save_path, 'wb') as f:
    pickle.dump([data_graph, padded_charges], f)
