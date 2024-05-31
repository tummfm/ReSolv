import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

from jax import config
config.update("jax_debug_nans", True)

import numpy as onp
import jax.numpy as jnp, jax
import chemtrain.amber_utils_qm7x as au
import chemtrain.nequip_amber_utils_qm7x as au_nequip

# Load data - alreaday shuffled
all_size = 10
energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")[:all_size]
pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[:all_size]
pad_forces = onp.load("QM7x_DB/shuffled_atom_forces_QM7x.npy")[:all_size]
pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[:all_size]
mol_id_data = onp.load("QM7x_DB/shuffled_atom_molecule_QM7x.npy")[:all_size]
mol_id_data = onp.array(mol_id_data, dtype=int)

for i in range(all_size):
    # pad_pos_temp =pad_pos[i]
    # pad_pos_500 = pad_pos_temp + 500
    x_spacing = 0
    pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])
    for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
        x_spacing += 6.0
        pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing

    pad_pos_temp = pad_pos[i]

    mol_id_data_temp = mol_id_data[i]
    mol_id_data_temp = onp.array(mol_id_data_temp, dtype=int)

    amber_init = au_nequip.build_amber_energy_fn(
            "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPFiles/mol_"+str(mol_id_data_temp)+"_AC.prmtop", unit_box_size=1000)

    value, grad = jax.value_and_grad(amber_init)(pad_pos_temp)
    print("True energy: ", energies[i])
    print("Value: ", value)
    print("Grad: ", grad)

# pred_energy_with_padding = amber_init(pad_pos_50)
#
# pred_energy_without_padding = amber_init(pad_pos_50[:13])
# print("Predicted energy with padding: ", pred_energy_with_padding)
# print("Predicted energy without padding: ", pred_energy_without_padding)




