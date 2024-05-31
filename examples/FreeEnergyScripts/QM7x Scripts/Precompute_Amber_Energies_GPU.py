import sys
import os

visible_device = ''
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import datetime
import numpy as onp
import jax, jax.numpy as jnp, jax.lax as lax
from jax_md import space

from chemtrain.jax_md_mod import custom_space
import chemtrain.copy_nequip_amber_utils_qm7x as au_nequip


load_energies = onp.load("QM7x_DB/atom_energies_QM7x.npy")
all_size = load_energies.shape[0]
# all_size = 100
min = int(sys.argv[1])
max = int(sys.argv[2])

# Load data - already shuffled
pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[0:all_size]
pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[0:all_size]

mol_id_data = onp.load("QM7x_DB/shuffled_atom_molecule_QM7x.npy")[0:all_size]
mol_id_data = onp.array(mol_id_data, dtype=int)

# Sorted mol_id_data
# Get indices of mol_id_data such that array is sorted ascending
mol_id_data_sorted_indices = onp.argsort(mol_id_data)
# Get indices of mol_id_data_sorted_indices such that mol_id_data is original shape again
# mol_id_data_sorted_indices_rev = onp.argsort(mol_id_data_sorted_indices)

pad_pos = pad_pos[mol_id_data_sorted_indices]
pad_species = pad_species[mol_id_data_sorted_indices]
mol_id_data = mol_id_data[mol_id_data_sorted_indices]

# Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
pad_species = onp.array(pad_species, dtype='int32')

# shrunk down data
pad_pos = pad_pos[min:max]
pad_species = pad_species[min:max]
mol_id_data = mol_id_data[min:max]


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


id_mapped = []
for id in mol_id_data:
    id_mapped.append(id_mapping[id])
id_mapped = jnp.array(id_mapped)

def energy_fn_template():
    amber_init = []
    for counter_map, mol_id in enumerate(list(id_mapping_rev.keys())):
        amber_fn = au_nequip.build_amber_energy_fn(
            "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPFiles/mol_" + str(
                id_mapping_rev[mol_id]) + "_AC.prmtop", displacement_fn)
        amber_init.append(amber_fn)
        print("Initialized amber number: {}", counter_map)

    def energy(pad_pos, idx):
        amber_prior_energy = lax.switch(idx, amber_init, pad_pos)
        return amber_prior_energy

    return energy

def update_template():
    energy_fn = energy_fn_template()

    def update(pad_pos, idx):
        energy, force = jax.value_and_grad(energy_fn)(pad_pos, idx)
        return energy, force
    return update


update = update_template()
update = jax.jit(update)
amber_energy_list = []
amber_force_list = []
# Get time stamp
start_time = datetime.datetime.now()
for i, idx in enumerate(id_mapped[:all_size]):
    print("Count: ", i)
    energy, force = update(pad_pos[i], id_mapped[i])
    amber_energy_list.append(energy)
    amber_force_list.append(force)
end_time = datetime.datetime.now()
print("Timed for 10k samples [s]: ", (end_time - start_time).total_seconds())
onp.save("QM7x_DB/shuffled_amber_energies_QM7x_"+str(min)+"_to_"+str(max)+".npy", onp.array(amber_energy_list))
onp.save("QM7x_DB/shuffled_amber_forces_QM7x_"+str(min)+"_to_"+str(max)+".npy", onp.array(amber_force_list))