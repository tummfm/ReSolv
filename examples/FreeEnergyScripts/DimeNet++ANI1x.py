import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

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
    load_dataset_name = 'ANI1xDB/'
    model = "ANI-1x"
    all_size = 4956005
    # all_size = 100
    # all_size = 100287   # First size of > 100.000 summing up combined configurations
    train_size = int(all_size * 0.7)

    batch_size = 3
    num_epochs = 2
    num_transition_steps = int(train_size * num_epochs * (1 / batch_size))
    batch_per_device = 3
    batch_cache = 3
    check_freq = None
    checkpoint = None
    seed_set = 1
    ID = "DimeNet_ID_A3"
    train_on = "EnergiesAndForces"


    # save_path = "savedTrainers/120422_Trainer_ANI1x_"+str(all_size)+"Samples_" + str(
    #     num_epochs) + "epochs_gu2emin7_gf1emin3_LRdecay1emin3_EnergiesAndForces.pkl"
    save_name = "170422_"+str(ID)+"_Trainer_ANI1x_"+str(all_size)+"Samples_" + str(
        num_epochs) + "epochs_gu2emin7_gf1emin3_LRdecay1emin3_"+train_on

    # Load data
    energies = onp.load(load_dataset_name+'energies.npy')[:all_size]
    pad_pos = onp.load(load_dataset_name+'pad_pos.npy')[:all_size]
    pad_forces = onp.load(load_dataset_name+'pad_forces.npy')[:all_size]
    pad_species = onp.load(load_dataset_name+'mask_species.npy')[:all_size]


    # TODO: Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops. Add uin8 there.
    pad_species = onp.array(pad_species, dtype='int32')

    # Convert forces and energies [ANI paper uses energy [Ha], forces [Ha/Å]], pos [Å] -> [nm]
    energies = energies * 2625.4996395  # [Ha] -> [kJ/mol]
    pad_forces = pad_forces * 26254.996395  # [Ha/Å] -> [kJ/mol*nm]
    pad_pos *= 0.1

    # Optimizer setup
    if model == 'ANI-1x':
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
    print("Done")
    # Create 100nm^3 box
    box = jnp.eye(3)*100

    # Compute max edges and angles to make training faster with PrecomputeEdgesAngles.py script.
    max_edges = 1608  # From precomputation
    max_angles = 48824  # From precomputation
    print("max_edges: ", max_edges)
    print("max_angles: ", max_angles)


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
                                           gamma_f=1e-4, gamma_u=2e-7, batch_per_device=batch_per_device, batch_cache=batch_cache)

    if checkpoint is not None:  # restart from a previous checkpoint
        trainer = util.load_trainer(checkpoint)

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
    onp.save('Train_val_test_data/pad_pos_train_seed'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_pos_train)
    onp.save('Train_val_test_data/pad_pos_val_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_pos_val)
    onp.save('Train_val_test_data/pad_pos_test_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_pos_test)

    onp.save('Train_val_test_data/energies_train_'+str(seed_set)+'_size'+str(all_size)+'.npy', energies_train)
    onp.save('Train_val_test_data/energies_val_'+str(seed_set)+'_size'+str(all_size)+'.npy', energies_val)
    onp.save('Train_val_test_data/energies_test_'+str(seed_set)+'_size'+str(all_size)+'.npy', energies_test)

    onp.save('Train_val_test_data/pad_forces_train_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_forces_train)
    onp.save('Train_val_test_data/pad_forces_val_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_forces_val)
    onp.save('Train_val_test_data/pad_forces_test_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_forces_test)

    onp.save('Train_val_test_data/pad_species_train_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_species_train)
    onp.save('Train_val_test_data/pad_species_val_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_species_val)
    onp.save('Train_val_test_data/pad_species_test_'+str(seed_set)+'_size'+str(all_size)+'.npy', pad_species_test)

    # trainer_FM_wrapper = trainers.wrapper_ForceMatching(trainerFM=trainer)

    print("Start training")
    # trainer_FM_wrapper.train(num_epochs, save_path=save_name)
    trainer.train(max_epochs=2)