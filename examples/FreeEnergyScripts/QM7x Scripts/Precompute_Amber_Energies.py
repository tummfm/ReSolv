import sys
import os

visible_device = ''
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import datetime
import jax, jax.numpy as jnp, jax.lax as lax
from jax_md import space

from chemtrain.jax_md_mod import custom_space
import chemtrain.copy_nequip_amber_utils_qm7x as au_nequip

min = int(sys.argv[1])
max = int(sys.argv[2])


load_energies = onp.load("QM7x_DB/atom_energies_QM7x.npy")
all_size = load_energies.shape[0]

# Load data - already shuffled
pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[min:max]
pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[min:max]

mol_id_data = onp.load("QM7x_DB/shuffled_atom_molecule_QM7x.npy")[min:max]
mol_id_data = onp.array(mol_id_data, dtype=int)

# Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
pad_species = onp.array(pad_species, dtype='int32')


# Create new mol_id_data with increasing numbers
unique_mol_ids = onp.unique(mol_id_data)
# Create a mapping from old to new mol_id
id_mapping = {}
id_mapping_rev = {}
for count, mol in enumerate(unique_mol_ids):
    id_mapping[mol] = count
    id_mapping_rev[count] = mol

# Scale padded positions
for i, pos in enumerate(pad_pos):
    pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])
for i, pos in enumerate(pad_pos):
    x_spacing = 0
    for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
        x_spacing += 20.0
        pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
box = jnp.eye(3) * 1000
scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
pad_pos = lax.map(scale_fn, pad_pos)

# Initialized amber energy functions
displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

amber_init = []
for counter_map, mol_id in enumerate(list(id_mapping_rev.keys())):
    amber_fn = au_nequip.build_amber_energy_fn(
        "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPFiles/mol_" + str(
            id_mapping_rev[mol_id]) + "_AC.prmtop", displacement_fn)
    # amber_fn = au_nequip.build_amber_energy_fn(
    #     "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPFiles/mol_" + str(
    #         id_mapping_rev[mol_id]) + "_AC.prmtop", unit_box_size=1000)
    amber_init.append(amber_fn)
    print("Initialized amber number: {}", counter_map)

amber_energy_list = []
amber_force_list = []
time_1 = datetime.datetime.now()
for count, id in enumerate(mol_id_data):
    # amber_energy = amber_init[id_mapping[id]](pad_pos[count])
    amber_energy, neg_amber_force = jax.value_and_grad(amber_init[id_mapping[id]])(pad_pos[count])
    amber_energy_list.append(amber_energy)
    amber_force_list.append(-neg_amber_force)

    if count % 1000 == 0:
        time_2 = datetime.datetime.now()
        print("{} samples, took {} s".format(count, (time_2 - time_1).total_seconds()))

# onp.save('check_energies.npy', onp.array(amber_energy_list))
# onp.save('check_forces.npy', onp.array(amber_force_list))

onp.save("QM7x_DB/shuffled_amber_energies_QM7x_"+str(min)+"_to_"+str(max)+".npy", onp.array(amber_energy_list))
onp.save("QM7x_DB/shuffled_amber_forces_QM7x_"+str(min)+"_to_"+str(max)+".npy", onp.array(amber_force_list))

# id_mapped = []
# for id in mol_id_data:
#     id_mapped.append(id_mapping[id])
# id_mapped = jnp.array(id_mapped)
#
# @jax.jit
# def update(i, idx, pad_pos, amber_initialized):
#     amber_energy, amber_neg_forces = jax.value_and_grad(amber_initialized)(pad_pos)
#     log['amber_energies'].at[i].set(amber_energy)
#     log['amber_forces'].at[i].set(-amber_neg_forces)
#
#
# log = {
#     'amber_energies': jnp.zeros(load_energies.shape),
#     'amber_forces': jnp.zeros(pad_pos.shape)
# }
#
# for i, idx in enumerate(id_mapped):
#     print("Count: ", i)
#     update(i, idx, pad_pos[i], amber_init[i])
