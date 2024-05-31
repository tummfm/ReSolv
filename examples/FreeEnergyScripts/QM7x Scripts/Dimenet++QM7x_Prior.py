import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

from jax import config
# config.update("jax_debug_nans", True)
import numpy as onp
import optax
from jax_md import space
from jax import random, numpy as jnp, lax


from util import Initialization
from chemtrain import trainers, util
from chemtrain.jax_md_mod import custom_space

# GENERAL NOTES: Learn a DimeNet++ potential based on ANI1x dataset

if __name__ == '__main__':
    # Choose setup
    load_energies = onp.load("QM7x_DB/atom_energies_QM7x.npy")
    all_size = load_energies.shape[0]
    model = "QM7x_prior"

    # all_size = 10

    train_size = int(all_size * 0.7)


    batch_size = 5
    num_epochs = 2
    num_transition_steps = int(train_size * num_epochs * (1 / batch_size))
    batch_per_device = 5
    batch_cache = 5

    ID = "DimeNet_ID_QM7x_5"
    train_on = "EnergiesAndForces"

    save_name = "090522_"+str(ID)+"_Trainer_ANI1x_"+str(all_size)+"Samples_" + str(
        num_epochs) + "epochs_gu2emin7_gf1emin3_LRdecay1emin3_"+train_on

    # Load data - alreaday shuffled
    energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")[:all_size]
    pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[:all_size]
    pad_forces = onp.load("QM7x_DB/shuffled_atom_forces_QM7x.npy")[:all_size]
    pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[:all_size]
    # TODO - once new preprocessing dataset is available add shuffle in fornt of atom_molecule_QM7x
    mol_id_data = onp.load("QM7x_DB/shuffled_atom_molecule_QM7x.npy")[:all_size]
    # energies = onp.load("QM7x_DB/atom_energies_QM7x.npy")[:all_size]
    # pad_pos = onp.load("QM7x_DB/atom_positions_QM7x.npy")[:all_size]
    # pad_forces = onp.load("QM7x_DB/atom_forces_QM7x.npy")[:all_size]
    # pad_species = onp.load("QM7x_DB/atom_numbers_QM7x.npy")[:all_size]
    # # TODO - once new preprocessing dataset is available add shuffle in fornt of atom_molecule_QM7x
    # mol_id_data = onp.load("QM7x_DB/atom_molecule_QM7x.npy")[:all_size]
    mol_id_data = onp.array(mol_id_data, dtype=int)
    # TODO: Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops. Add uin8 there.
    pad_species = onp.array(pad_species, dtype='int32')

    # Unpad for debugging
    # nonpadded_num = onp.count_nonzero(pad_species[0])
    # pad_pos = pad_pos[:, :nonpadded_num]
    # pad_forces = pad_forces[:, :nonpadded_num]
    # pad_species = pad_species[:, :nonpadded_num]


    # Not needed anymore
    # Remove data with mol_id == 2066 as Anton mentioned that this does not have a topology file
    # remove_entries_list = [pos for pos, i in enumerate(mol_id_data) if i == 2066]
    # if len(remove_entries_list) > 0:
    #     energies = onp.delete(energies, remove_entries_list)
    #     pad_pos = onp.delete(pad_pos, remove_entries_list)
    #     pad_forces = onp.delete(pad_forces, remove_entries_list)
    #     pad_species = onp.delete(pad_species, remove_entries_list)
    #     mol_id_data = onp.delete(mol_id_data, remove_entries_list)


    # Create new mol_id_data with increasing numbers
    unique_mol_ids = onp.unique(mol_id_data)
    # new_mol_ids = onp.arange(len(unique_mol_ids))

    # Create a mapping from old to new mol_id
    id_mapping = {}
    id_mapping_rev = {}
    for count, mol in enumerate(unique_mol_ids):
        id_mapping[mol] = count
        id_mapping_rev[count] = mol


    mol_id_data_new = [id_mapping[i] for i in mol_id_data]

    # mol_id_data_new = onp.arange(len(mol_id_data))

    # Convert forces and energies - QM7x paper uses energy [eV], forces [eV/Å]], pos [Å]
    energies = energies * 96.49  # [eV] -> [kJ/mol]
    pad_forces = pad_forces * 964.9  # [eV/Å] -> [kJ/mol*nm]
    pad_pos *= 0.1   # [Å] to [nm]

    # Optimizer setup
    if model == 'QM7x_prior':
        initial_lr = 0.001
    else:
        raise NotImplementedError

    lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, 0.001)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule)
    )

    # Take padded positions.
    #  1. Add to non zero values 50nm
    #  2. Set padded pos at x=0.6, 1.2, 1.8, .. [nm], y = z = 0. -> Energy contribution is zero

    for i, pos in enumerate(pad_pos):
        pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])
    for i, pos in enumerate(pad_pos):
        x_spacing = 0
        for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
            x_spacing += 0.6
            pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
    box = jnp.eye(3)*100

    # Compute max edges and angles to make training faster with PrecomputeEdgesAngles.py script.
    max_edges = 496  # From precomputation
    max_angles = 10216  # From precomputation
    print("max_edges: ", max_edges)
    print("max_angles: ", max_angles)


    scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
    pad_pos = lax.map(scale_fn, pad_pos)

    # Turn all arrays to jax.numpy format
    energies = jnp.array(energies)
    pad_forces = jnp.array(pad_forces)
    pad_species = jnp.array(pad_species)
    pad_pos = jnp.array(pad_pos)
    mol_id_data_new = jnp.array(mol_id_data_new)

    # TODO - Pick R_init, spec_init. Be aware that in select_model() neighborlist needs to be handcrafted on first molecule!!!
    R_init = pad_pos[0]
    spec_init = pad_species[0]

    # Initial values
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
    model_init_key = random.PRNGKey(0)

    energy_fn_template, _, init_params, nbrs_init = Initialization.select_model(model=model, init_pos=R_init,
                                                                                displacement=displacement,
                                                                                box=box, model_init_key=model_init_key,
                                                                                species=spec_init, mol_id_data=mol_id_data,
                                                                                id_mapping_rev=id_mapping_rev)

    # TODO Rethink the energy / force weighting. Should it be equally sized now?
    trainer = trainers.ForceMatching_QM7x(init_params=init_params, energy_fn_template=energy_fn_template,
                                           nbrs_init=nbrs_init, optimizer=optimizer, position_data=pad_pos,
                                           species_data=pad_species, mol_id_data=mol_id_data,  energy_data=energies, force_data=pad_forces,
                                           gamma_f=1e-4, gamma_u=1e-4, batch_per_device=batch_per_device,
                                           batch_cache=batch_cache)

    trainer_FM_wrapper = trainers.wrapper_ForceMatching(trainerFM=trainer)
    trainer_FM_wrapper.train(num_epochs, save_path=save_name)
