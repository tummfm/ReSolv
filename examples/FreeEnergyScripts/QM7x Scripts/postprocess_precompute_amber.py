import sys
import os

visible_device = ''
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp

# Load sorted data
# Create a numpy array from 0 to 4195237 with step size 10000
x = onp.arange(0, 4195237, 10000)
x = onp.append(x, 4195237)
energy_list = []
force_list = []
for i in range(len(x) - 1):
    energies = onp.load("QM7x_DB/shuffled_amber_energies_QM7x_"+str(x[i])+"_to_"+str(x[i+1])+".npy")
    forces = onp.load("QM7x_DB/shuffled_amber_forces_QM7x_"+str(x[i])+"_to_"+str(x[i+1])+".npy")
    energy_list.append(energies)
    force_list.append(forces)
# Concatenate first dimension of list
energy_list = onp.concatenate(energy_list)
force_list = onp.concatenate(force_list)

# Shuffle the energies and forces
mol_id_data = onp.load("QM7x_DB/shuffled_atom_molecule_QM7x.npy")
mol_id_data = onp.array(mol_id_data, dtype=int)
# Get indices of mol_id_data such that array is sorted ascending
mol_id_data_sorted_indices = onp.argsort(mol_id_data)
# Get indices of mol_id_data_sorted_indices such that mol_id_data is original shape again
mol_id_data_sorted_indices_rev = onp.argsort(mol_id_data_sorted_indices)
energy_list = energy_list[mol_id_data_sorted_indices_rev]
force_list = force_list[mol_id_data_sorted_indices_rev]

# Save shuffled energies and forces
onp.save("QM7x_DB/shuffled_amber_energies_QM7x_0_to_4195237.npy", energy_list)
onp.save("QM7x_DB/shuffled_amber_forces_QM7x_0_to_4195237.npy", force_list)

