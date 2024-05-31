import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp


# Load data
load_energies = onp.load("QM7x_DB/atom_energies_QM7x.npy")
all_size = load_energies.shape[0]


# Load data - alreaday shuffled
energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")[:all_size]
pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[:all_size]
pad_forces = onp.load("QM7x_DB/shuffled_atom_forces_QM7x.npy")[:all_size]
pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[:all_size]

# ******* Manual settings *********


def precompute_avg_neighbors(pad_pos):
    """Precompute average neighbors, relevant for Nequip normalization."""
    pad_pos *= 0.1
    neighbors_list = []
    for h, pos in enumerate(pad_pos):
        num = int(onp.count_nonzero(pos) / 3)
        for i in range(num):
            temp_neighbors = 0
            for j in range(num):
                if i != j:
                    dist = onp.linalg.norm(pos[i] - pos[j])
                    if dist <= 0.5:
                        temp_neighbors += 1
            neighbors_list.append(temp_neighbors)
        print('Atom {} done'.format(h))
    avg_nbrs = int(onp.ceil(onp.mean(neighbors_list)))
    print('Neighborlist: ', neighbors_list)
    print('Averge neighbors: ', avg_nbrs)
    return 0

def precompute_energy_shift(energies):
    """Precompute energy shift."""
    return onp.mean(energies)


def precompute_rms_force(forces):
    """Compute root mean sqaure forces for the scaling."""
    count_forces = 0
    sum_forces = 0
    for single_force in forces:
        num_forces = int(onp.count_nonzero(single_force) / 3)
        single_force = single_force[:num_forces]**2
        sum_forces += onp.sum(single_force)
        count_forces += num_forces * 3

    return onp.sqrt(sum_forces / count_forces)
if __name__ == '__main__':

    precompute_avg_neighbors(pad_pos=pad_pos)
    shift_factor = precompute_energy_shift(energies=energies)
    scaling_factor = precompute_rms_force(forces=pad_forces)
    print("Shift factor: ", shift_factor)
    print("Scaling factor: ", scaling_factor)