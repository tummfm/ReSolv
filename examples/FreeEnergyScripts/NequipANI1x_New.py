import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Avoid error in jax 0.4.25
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import numpy as onp
import optax
from jax_md import space
from jax import random, numpy as jnp, lax


from util import Initialization
from chemtrain import trainers, util
from chemtrain.jax_md_mod import custom_space

# GENERAL NOTES: Learn a Nequip potential based on ANI1x dataset

if __name__ == '__main__':
    # Choose setup
    load_dataset_name = 'ANI1xDB/'
    model = "NequIP"
    all_size = 4956005
    # all_size = 100287   # First size of > 100.000 summing up combined configurations
    train_size = int(all_size * 0.7)

    batch_size = 5
    num_epochs = 2
    num_transition_steps = int(train_size * num_epochs * (1 / batch_size))
    batch_per_device = 5
    batch_cache = 5
    check_freq = None

    nequip_units = True
    seed_set = 1
    id_num = "Nequip_ID_A1"
    lr_decay = 1e-3
    shift_b = "False"
    scale_b = "False"
    mlp = "4"
    train_on = "EnergiesAndForces"

    save_path = "savedTrainers/170422_Nequip_"+id_num+"_Trainer_ANI1x_Energies"+str(all_size)+"samples_" + str(
        num_epochs) + "epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp"+mlp+"_Shift"+shift_b+"_Scale"+scale_b+"_"+train_on+".pkl"
    save_name = "170422_Nequip_"+id_num+"_Trainer_ANI1x_Energies"+str(all_size)+"samples_" + str(
        num_epochs) + "epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp"+mlp+"_Shift"+shift_b+"_Scale"+scale_b+"_"+train_on

    # Load data
    energies = onp.load(load_dataset_name+'energies.npy')[:all_size]
    pad_pos = onp.load(load_dataset_name+'pad_pos.npy')[:all_size]
    pad_forces = onp.load(load_dataset_name+'pad_forces.npy')[:all_size]
    pad_species = onp.load(load_dataset_name+'mask_species.npy')[:all_size]

    # num_sub = onp.linspace(0, all_size-1, 2000, dtype=int)
    # energies = energies[num_sub]
    # pad_pos = pad_pos[num_sub]
    # pad_forces = pad_forces[num_sub]
    # pad_species = pad_species[num_sub]

    # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
    pad_species = onp.array(pad_species, dtype='int32')

    # Convert forces and energies [ANI paper uses energy [Ha], forces [Ha/Å]], pos [Å] -> [nm]
    energies = energies * 2625.4996395  # [Ha] -> [kJ/mol]
    pad_forces = pad_forces * 26254.996395  # [Ha/Å] -> [kJ/mol*nm]
    pad_pos *= 0.1

    # # Convert energies, forces, positions to eV, Å
    if nequip_units:
        energies *= 0.01036  # [kJ/mol] to [eV]
        pad_forces *= 0.001036  # [kJ/mol*mn] to [eV/Å]
        pad_pos *= 10

    # Optimizer setup
    if model == 'NequIP':
        initial_lr = 0.01
    else:
        raise NotImplementedError

    # lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, 0.001)
    lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, lr_decay)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule)
    )

    # Take padded positions.
    #  1. Add to non zero values 50nm
    #  2. Set padded pos at x=0.6, 1.2, 1.8, .. [nm], y = z = 0. -> Energy contribution is zero

    print("Start padding")
    if not nequip_units:
        for i, pos in enumerate(pad_pos):
            pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])
        for i, pos in enumerate(pad_pos):
            x_spacing = 0
            for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                x_spacing += 0.6
                pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
        print("Done")
        # Create 100nm^3 box
        box = jnp.eye(3)*100
    elif nequip_units:
        for i, pos in enumerate(pad_pos):
            pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])
        for i, pos in enumerate(pad_pos):
            x_spacing = 0
            for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                x_spacing += 6.0
                pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
        print("Done")
        # Create 100nm^3 box
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

    # Pick R_init, spec_init. Be aware that in select_model() neighborlist was handcrafted on first molecule!!!
    R_init = pad_pos[0]
    spec_init = pad_species[0]

    # Shuffle the data
    onp.random.seed(seed=seed_set)
    shuffle_indices = onp.arange(energies.shape[0])
    onp.random.shuffle(shuffle_indices)

    energies = energies[shuffle_indices]
    pad_pos = pad_pos[shuffle_indices]
    pad_forces = pad_forces[shuffle_indices]
    pad_species = pad_species[shuffle_indices]

    # Initial values
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
    model_init_key = random.PRNGKey(0)

    energy_fn_template, _, init_params, nbrs_init = Initialization.select_model(model=model, init_pos=R_init,
                                                                                displacement=displacement,
                                                                                box=box, model_init_key=model_init_key,
                                                                                species=spec_init)

    trainer = trainers.ForceMatching_ANI1x(init_params=init_params, energy_fn_template=energy_fn_template,
                                           nbrs_init=nbrs_init, optimizer=optimizer, position_data=pad_pos,
                                           species_data=pad_species, energy_data=energies, force_data=pad_forces,
                                           gamma_f=1e-1, gamma_u=2e-4, batch_per_device=batch_per_device, batch_cache=batch_cache)

    # Save the dataset for testing later - Do splitting same as in dataprocessing - test, val, testsplit
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    dataset_size = pad_pos.shape[0]
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)

    pad_pos_train = pad_pos[:train_size]
    pad_pos_val = pad_pos[train_size:train_size + val_size]
    pad_pos_test = pad_pos[train_size + val_size:]

    energies_train = energies[:train_size]
    energies_val = energies[train_size:train_size + val_size]
    energies_test = energies[train_size + val_size:]

    pad_forces_train = pad_forces[:train_size]
    pad_forces_val = pad_forces[train_size:train_size + val_size]
    pad_forces_test = pad_forces[train_size + val_size:]

    pad_species_train = pad_species[:train_size]
    pad_species_val = pad_species[train_size:train_size + val_size]
    pad_species_test = pad_species[train_size + val_size:]
    # Save data
    onp.save('Train_val_test_data/pad_pos_train_seed'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_pos_train)
    onp.save('Train_val_test_data/pad_pos_val_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_pos_val)
    onp.save('Train_val_test_data/pad_pos_test_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_pos_test)

    onp.save('Train_val_test_data/energies_train_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', energies_train)
    onp.save('Train_val_test_data/energies_val_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', energies_val)
    onp.save('Train_val_test_data/energies_test_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', energies_test)

    onp.save('Train_val_test_data/pad_forces_train_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_forces_train)
    onp.save('Train_val_test_data/pad_forces_val_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_forces_val)
    onp.save('Train_val_test_data/pad_forces_test_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_forces_test)

    onp.save('Train_val_test_data/pad_species_train_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_species_train)
    onp.save('Train_val_test_data/pad_species_val_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_species_val)
    onp.save('Train_val_test_data/pad_species_test_'+str(seed_set)+'_size'+str(all_size)+'_'+train_on+'.npy', pad_species_test)

    trainer_FM_wrapper = trainers.wrapper_ForceMatching(trainerFM=trainer)
    print("Start training")
    trainer_FM_wrapper.train(num_epochs, save_path=save_name)