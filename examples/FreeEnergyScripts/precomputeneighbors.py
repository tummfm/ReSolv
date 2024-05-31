import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp


# Load data
# all_size = 100287
all_size = 4956005
# all_size = 300
load_dataset_name = 'ANI1xDB/'
energies = onp.load(load_dataset_name + 'energies.npy')[:all_size]
pad_pos = onp.load(load_dataset_name + 'pad_pos.npy')[:all_size]
pad_forces = onp.load(load_dataset_name + 'pad_forces.npy')[:all_size]
pad_species = onp.load(load_dataset_name + 'mask_species.npy')[:all_size]

# ******* Manual settings *********
use_nequip__units = False

# # choose a 2k subset of the data
# num_sub = onp.linspace(0, all_size - 1, 2000, dtype=int)
# energies = energies[num_sub]
# pad_pos = pad_pos[num_sub]
# pad_forces = pad_forces[num_sub]
# pad_species = pad_species[num_sub]

if use_nequip__units:
    energies *= 0.01036  # [kJ/mol] to [eV]
    pad_forces *= 0.001036  # [kJ/mol*mn] to [eV/Å]
    pad_pos *= 10

def precompute_avg_neighbors(pad_pos):
    """Precompute average neighbors, relevant for Nequip normalization."""
    pad_pos *= 0.1
    counter = 0
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