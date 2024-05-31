import copy

import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Avoid error in jax 0.4.25
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

from jax import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
import numpy as onp
import optax
from jax_md import space
from jax import random, numpy as jnp, lax


from util import Initialization
from chemtrain import trainers, util
from chemtrain.jax_md_mod import custom_space

# GENERAL NOTES: Learn a Nequip potential based onQM7x dataset

if __name__ == '__main__':
    # Choose setup
    # model = "NequIP_QM7x_prior"
    model = "NequIP_QM7x_priorInTarget"
    check_freq = None
    use_heatup = False
    use_clipping = False
    scale_target = True    # Scale target energies & forces
    scale_amber = False     # Scale amber energies & forces
    scale_target_energies = False     # Scale energies
    scale_target_forces = False       # Scale forces
    remove_100k_largest_amber = True     # Remove 10k largest amber energies
    remove_500k_largest_amber = False
    remove_100k_largest_amber_forces = False
    keep_100k_smallest_amber = False
    shift_and_scale_per_atom = False
    use_only_equilibrium = False
    amber_zero = False
    subset_equilibrium = False
    shift_and_scale_per_atom_amber = False

    test_size = 10100
    val_ratio = 0.1
    train_ratio = 1 - 0.1 - 0.0024073

    batch_size = 5
    num_epochs = 8
    # date = "171023"
    # date = "261023"
    # date = "181123"
    date = "281123"
    nequip_units = True

    # Optimizer setup
    lr_decay = 1e-4
    initial_lr = 1e-3

    if use_only_equilibrium:
        load_energies = onp.load("QM7x_DB/equilibrium_shuffled_atom_energies_QM7x.npy")
    else:
        load_energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")
        # load_energies = onp.load("QM7x_DB/atom_energies_QM7x_1000.npy")


    # num_warmup_steps = int(train_size * 1 * (1 / batch_size))
    batch_per_device = 5
    batch_cache = 100

    # id_num = "Nequip_QM7x_All_8epochs_iL5emin3_lrdecay5emin4_scaledTargets_LargeTrainingSet_Cutoff5A"
    id_num = "Nequip_QM7x_All_WithAmber_"+str(num_epochs)+"epochs_iL"+str(initial_lr)+"_lrdecay"+str(lr_decay)+"_scaledTargets_" \
             "LargeTrainingSet_Cutoff4A"

    shift_b = "False"
    scale_b = "False"
    mlp = "4"
    train_on = "EnergiesAndForces"

    save_path = "savedTrainers/"+date+"_QM7x_Nequip_"+id_num+"_Trainer_ANI1x_Energies" + str(
        num_epochs) + "epochs_mlp"+mlp+"_Shift"+shift_b+"_Scale"+scale_b+"_"+train_on+".pkl"
    save_name = date+"_QM7x_Nequip_"+id_num+"_Trainer_ANI1x_Energies"+ str(
        num_epochs) + "epochs_mlp"+mlp+"_Shift"+shift_b+"_Scale"+scale_b+"_"+train_on

    if use_only_equilibrium:
        # Load data - already shuffled
        energies = onp.load("QM7x_DB/equilibrium_shuffled_atom_energies_QM7x.npy")[:all_size]
        pad_pos = onp.load("QM7x_DB/equilibrium_shuffled_atom_positions_QM7x.npy")[:all_size]
        pad_forces = onp.load("QM7x_DB/equilibrium_shuffled_atom_forces_QM7x.npy")[:all_size]
        pad_species = onp.load("QM7x_DB/equilibrium_shuffled_atom_numbers_QM7x.npy")[:all_size]

        amber_energy = onp.zeros_like(energies)
        amber_force = onp.zeros_like(pad_forces)

        mol_id_data = None
        id_mapping_rev = None
    else:
        # Load data - already shuffled
        energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")
        pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")
        pad_forces = onp.load("QM7x_DB/shuffled_atom_forces_QM7x.npy")
        pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")

        amber_energy = onp.load("QM7x_DB/shuffled_amber_energies_QM7x_0_to_4195237.npy")
        amber_force = onp.load("QM7x_DB/shuffled_amber_forces_QM7x_0_to_4195237.npy")

        mol_id_data = onp.load("QM7x_DB/shuffled_atom_molecule_QM7x.npy")
        mol_id_data = onp.array(mol_id_data, dtype=int)

        id_mapping_rev = None

    if scale_amber:
        # Scaling amber energies and forces
        amber_energy /= 1000
        amber_force /= 1000

    if remove_500k_largest_amber:
        # Remove 10k largest amber energies
        remove_max_amber_energy_indices = onp.argsort(amber_energy)[-500000:]
        max_amber_forces = onp.array([onp.max(onp.abs(force)) for force in amber_force])
        remove_max_amber_forces_indices = onp.argsort(max_amber_forces)[-500000:]

        # Check how many indices are the same in both lists
        print("Number of indices that are the same in both lists: ", len(onp.intersect1d(remove_max_amber_energy_indices, remove_max_amber_forces_indices)))


        # Combine the two lists and remove duplicates
        remove_max_amber_indices = onp.unique(onp.concatenate((remove_max_amber_energy_indices, remove_max_amber_forces_indices), axis=0))
        print("Length of remove_max_amber_indices: ", len(remove_max_amber_indices))
        print("Debug")

        # if assert not true print error
        if not amber_energy.shape[0] == energies.shape[0]:
            print("ERROR: amber_energy.shape[0] != energies.shape[0]")
            sys.exit()

        # Remove the indices from the data
        energies = onp.delete(energies, remove_max_amber_indices, axis=0)
        pad_pos = onp.delete(pad_pos, remove_max_amber_indices, axis=0)
        pad_forces = onp.delete(pad_forces, remove_max_amber_indices, axis=0)
        pad_species = onp.delete(pad_species, remove_max_amber_indices, axis=0)
        amber_energy = onp.delete(amber_energy, remove_max_amber_indices, axis=0)
        amber_force = onp.delete(amber_force, remove_max_amber_indices, axis=0)
        mol_id_data = onp.delete(mol_id_data, remove_max_amber_indices, axis=0)


        new_all_size = energies.shape[0]
        train_size = int(new_all_size * train_ratio)
        num_transition_steps = int(train_size * num_epochs * (1 / batch_size))
        num_warmup_steps = int(train_size * 1 * (1 / batch_size))

    if remove_100k_largest_amber:
        # Remove 10k largest amber energies
        remove_max_amber_energy_indices = onp.argsort(amber_energy)[-100000:]
        max_amber_forces = onp.array([onp.max(onp.abs(force)) for force in amber_force])
        remove_max_amber_forces_indices = onp.argsort(max_amber_forces)[-100000:]

        # Check how many indices are the same in both lists
        print("Number of indices that are the same in both lists: ", len(onp.intersect1d(remove_max_amber_energy_indices, remove_max_amber_forces_indices)))


        # Combine the two lists and remove duplicates
        remove_max_amber_indices = onp.unique(onp.concatenate((remove_max_amber_energy_indices, remove_max_amber_forces_indices), axis=0))
        print("Length of remove_max_amber_indices: ", len(remove_max_amber_indices))
        print("Debug")

        # if assert not true print error
        if not amber_energy.shape[0] == energies.shape[0]:
            print("ERROR: amber_energy.shape[0] != energies.shape[0]")
            sys.exit()

        # Remove the indices from the data
        energies = onp.delete(energies, remove_max_amber_indices, axis=0)
        pad_pos = onp.delete(pad_pos, remove_max_amber_indices, axis=0)
        pad_forces = onp.delete(pad_forces, remove_max_amber_indices, axis=0)
        pad_species = onp.delete(pad_species, remove_max_amber_indices, axis=0)
        amber_energy = onp.delete(amber_energy, remove_max_amber_indices, axis=0)
        amber_force = onp.delete(amber_force, remove_max_amber_indices, axis=0)
        mol_id_data = onp.delete(mol_id_data, remove_max_amber_indices, axis=0)

        new_all_size = energies.shape[0]
        train_size = int(new_all_size * train_ratio)
        num_transition_steps = int(train_size * num_epochs * (1 / batch_size))

    if remove_100k_largest_amber_forces:
        max_amber_forces = onp.array([onp.max(onp.abs(force)) for force in amber_force])
        remove_max_amber_forces_indices = onp.argsort(max_amber_forces)[-100000:]

        # Combine the two lists and remove duplicates
        remove_max_amber_indices = remove_max_amber_forces_indices

        # Remove the indices from the data
        energies = onp.delete(energies, remove_max_amber_indices, axis=0)
        pad_pos = onp.delete(pad_pos, remove_max_amber_indices, axis=0)
        pad_forces = onp.delete(pad_forces, remove_max_amber_indices, axis=0)
        pad_species = onp.delete(pad_species, remove_max_amber_indices, axis=0)
        amber_energy = onp.delete(amber_energy, remove_max_amber_indices, axis=0)
        amber_force = onp.delete(amber_force, remove_max_amber_indices, axis=0)
        mol_id_data = onp.delete(mol_id_data, remove_max_amber_indices, axis=0)
        amber_energy = onp.zeros_like(amber_energy)

        new_all_size = energies.shape[0]
        train_size = int(new_all_size * 0.7)
        num_transition_steps = int(train_size * num_epochs * (1 / batch_size))

    if keep_100k_smallest_amber:
        idx = onp.argsort(amber_energy)[:100000]
        energies = energies[idx]
        pad_pos = pad_pos[idx]
        pad_forces = pad_forces[idx]
        pad_species = pad_species[idx]
        amber_energy = amber_energy[idx]
        amber_force = amber_force[idx]
        mol_id_data = mol_id_data[idx]

        new_all_size = energies.shape[0]
        train_size = int(new_all_size * 0.9874008)

    all_size = energies.shape[0]
    test_ratio = (test_size / all_size) - 1e-7
    train_ratio = 1 - test_ratio - val_ratio
    train_size = int(all_size * train_ratio)
    num_transition_steps = int(train_size * num_epochs * (1 / batch_size))
    print("Test size: ", test_size)
    if test_size != (all_size - int(train_ratio * all_size) - int(val_ratio * all_size)):
        print("ERROR: Test size does not match intended size")
        sys.exit()


    if scale_target:
        # Modify the target energies + forces by subtracting the amber contribution
        energies = energies - amber_energy
        pad_forces = pad_forces - amber_force

        # Do scaling of energies and forces
        # Compute the mean potential energy over the training dataset
        mean_energy = onp.mean(energies[:train_size])
        std_energy = onp.std(energies[:train_size])

        avg_force_rms = onp.mean([onp.sqrt(onp.mean(force_single ** 2)) for force_single in pad_forces[:train_size]])
        # Compute the root mean square of the force components over the training set
        # rms_force = onp.sqrt(onp.mean(pad_forces[:train_size] ** 2))

        energies = (energies - mean_energy) / avg_force_rms
        pad_forces = pad_forces / avg_force_rms

        # Set amber energies & forces to 0 for case with amber in target
        amber_energy = onp.zeros_like(amber_energy)
        amber_force = onp.zeros_like(amber_force)

        avg_std_force = 1.0
        mean_per_atom_energy = 0.0

    if amber_zero:
        amber_energy = onp.zeros_like(amber_energy)
        amber_force = onp.zeros_like(amber_force)

    if subset_equilibrium:
        energies = energies[:100]
        pad_pos = pad_pos[:100]
        pad_forces = pad_forces[:100]
        pad_species = pad_species[:100]
        amber_energy = amber_energy[:100]
        amber_force = amber_force[:100]
        train_size = int(energies.shape[0] * 0.7)
        num_transition_steps = int(train_size * num_epochs * (1 / batch_size))


    if shift_and_scale_per_atom:
        # Modify the target energies + forces by subtracting the amber contribution
        # energies = energies - amber_energy
        # pad_forces = pad_forces - amber_force

        num_atoms = onp.count_nonzero(pad_species[:train_size])
        sum_energy = onp.sum(energies[:train_size])

        # Compute scale and shift
        mean_per_atom_energy = sum_energy / num_atoms
        avg_std_force = onp.mean([onp.std(force_single) for force_single in pad_forces[:train_size]])

        amber_energy = onp.zeros_like(energies)
        amber_force = onp.zeros_like(pad_forces)

    if shift_and_scale_per_atom_amber:
        # Modify the target energies + forces by subtracting the amber contribution
        energies = energies - amber_energy
        pad_forces = pad_forces - amber_force

        num_atoms = onp.count_nonzero(pad_species[:train_size])
        sum_energy = onp.sum(energies[:train_size])

        # Compute scale and shift
        mean_per_atom_energy = sum_energy / num_atoms
        avg_std_force = onp.mean([onp.std(force_single) for force_single in pad_forces[:train_size]])

        amber_energy = onp.zeros_like(energies)
        amber_force = onp.zeros_like(pad_forces)

    # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
    pad_species = onp.array(pad_species, dtype='int32')

    # # Convert energies, forces, positions to eV, Angstrom
    if not nequip_units:
        # Convert forces and energies [QM7x paper uses energy [eV], forces [eV/Ang]], pos [Ang]
        energies = energies * 96.49  # [eV] -> [kJ/mol]
        pad_forces = pad_forces * 964.9  # [eV/Ang] -> [kJ/mol*nm]
        pad_pos *= 0.1  # [Ang] to [nm]

    # # Adam Belief optimizer
    lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, lr_decay)
    optimizer = optax.chain(
        optax.scale_by_belief(),
        optax.scale_by_schedule(lr_schedule)
    )

    epochs_all = num_epochs

    # Take padded positions.
    #  1. Add to non zero values 50nm
    #  2. Set padded pos at x=0.6, 1.2, 1.8, .. [nm], y = z = 0. -> Energy contribution is zero

    if not nequip_units:
        for i, pos in enumerate(pad_pos):
            pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])
        for i, pos in enumerate(pad_pos):
            x_spacing = 0
            for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                x_spacing += 0.6
                pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
        box = jnp.eye(3)*100

    elif nequip_units:
        for i, pos in enumerate(pad_pos):
            pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])
        for i, pos in enumerate(pad_pos):
            x_spacing = 0
            for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                x_spacing += 15.0
                pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
        box = jnp.eye(3)*1000
    else:
        raise NotImplementedError

    scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
    pad_pos = lax.map(scale_fn, pad_pos)

    # Turn all arrays to jax.numpy format
    energies = jnp.array(energies)
    pad_forces = jnp.array(pad_forces)
    pad_species = jnp.array(pad_species)
    pad_pos = jnp.array(pad_pos)

    amber_energy = jnp.array(amber_energy)
    amber_force = jnp.array(amber_force)

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

    if shift_and_scale_per_atom or shift_and_scale_per_atom_amber or scale_target:
        print("Shift and scale per atom or shift and scale per atom amber")
        trainer = trainers.ForceMatching_QM7x(init_params=init_params, energy_fn_template=energy_fn_template,
                                              nbrs_init=nbrs_init, optimizer=optimizer, position_data=pad_pos,
                                              species_data=pad_species, amber_energy_data=amber_energy,
                                              amber_force_data=amber_force, energy_data=energies, force_data=pad_forces,
                                              gamma_f=1, gamma_u=1e-2, batch_per_device=batch_per_device,
                                              batch_cache=batch_cache, train_ratio=train_ratio, val_ratio=val_ratio,
                                              scale_U_F=avg_std_force, shift_U_F=mean_per_atom_energy)
    else:
        raise NotImplementedError

    trainer_FM_wrapper = trainers.wrapper_ForceMatching(trainerFM=trainer)
    trainer_FM_wrapper.train(epochs_all, save_path=save_name)