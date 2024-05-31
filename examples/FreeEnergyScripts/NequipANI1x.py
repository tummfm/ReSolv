import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import optax
from jax_md import space
from jax import random, numpy as jnp, lax
import matplotlib.pyplot as plt

from util import Initialization
from chemtrain import trainers, util
from chemtrain.jax_md_mod import custom_space

# # Debug
# from jax.config import config
# config.update("jax_debug_nans", True)

if __name__ == '__main__':

    # 1. Load the dataset.
    # 2. Pad the dataset.

    # Choose setup
    # load_dataset_name = '../../../../FreeEnergySolubility/examples/FreeEnergyScripts/ANI1xDB/'
    load_dataset_name = 'ANI1xDB/'
    model = "ANI-1x"
    # all_size = 4956005
    all_size = 1000
    train_size = int(all_size * 0.7)

    batch_size = 5
    num_epochs = 70
    num_transition_steps = int(train_size * num_epochs * (1 / batch_size))
    batch_per_device = 5
    batch_cache = 10
    check_freq = None
    save_path = "ANI1xDB/170222_ANI1x_subset_First1000_"+str(num_epochs)+"epochs_Nequip.pkl"


    # Load data
    energies = onp.load(load_dataset_name+'energies.npy')[:all_size]
    pad_pos = onp.load(load_dataset_name+'pad_pos.npy')[:all_size]
    pad_forces = onp.load(load_dataset_name+'pad_forces.npy')[:all_size]
    pad_species = onp.load(load_dataset_name+'mask_species.npy')[:all_size]

    # Shuffle the data
    shuffle_indices = onp.arange(energies.shape[0])
    onp.random.shuffle(shuffle_indices)

    energies = energies[shuffle_indices]
    pad_pos = pad_pos[shuffle_indices]
    pad_forces = pad_forces[shuffle_indices]
    pad_species = pad_species[shuffle_indices]

    # # Debug - Undo padding to see result
    # pad_pos = pad_pos[:, :20, :]
    # pad_forces = pad_forces[:, :20, :]
    # pad_species = pad_species[:, :20]

    pad_species = onp.array(pad_species, dtype='int32')

    # Convert forces and energies [ANI paper uses energy [Ha], forces [Ha/Å]], pos [Å] -> [nm]
    energies = energies * 2625.4996395  # [Ha] -> [kJ/mol]
    pad_forces = pad_forces * 26254.996395  # [Ha/Å] -> [kJ/mol*nm]
    pad_pos *= 0.1

    # TODO: Set up new optimizer for Nequip
    # Optimizer setup
    if model == 'ANI-1x':
        initial_lr = 0.001
    else:
        raise NotImplementedError

    lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, 0.01)
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

    # Create fractional coordinates
    scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
    pad_pos = lax.map(scale_fn, pad_pos)

    # Turn all arrays to jax.numpy format
    energies = jnp.array(energies)
    pad_forces = jnp.array(pad_forces)
    pad_species = jnp.array(pad_species)
    pad_pos = jnp.array(pad_pos)


    # Initial values
    R_init = pad_pos[0]
    spec_init = pad_species[0]
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
    model_init_key = random.PRNGKey(0)

    energy_fn_template, neighbor_fn, init_params, nbrs_init = \
        Initialization.initialize_Nequip_model(species=spec_init, displacement=displacement,
                                               box=box, init_pos=R_init)

    # TODO: Current state
    # TODO: Missing above: optimizer

    #
    # # TODO: Set up new model in Initialization.select_model()
    # energy_fn_template, _, init_params, nbrs_init = Initialization.select_model(model=model, init_pos=R_init,
    #                                                                             displacement=displacement,
    #                                                                             box=box, model_init_key=model_init_key,
    #                                                                             species=spec_init)


    # TODO: Set up new Nequip trainer
    trainer = trainers.ForceMatching_ANI1x(init_params=init_params, energy_fn_template=energy_fn_template, nbrs_init=nbrs_init,
                                           optimizer=optimizer, position_data=pad_pos, species_data=pad_species,
                                           energy_data=energies, force_data=pad_forces, gamma_u=0.0001,
                                           batch_per_device=batch_per_device, batch_cache=batch_cache)


    # TODO: Training might take different arguments.
    trainer.train(num_epochs, checkpoint_freq=check_freq)
    trainer.save_trainer(save_path)


    # TODO: trainer might not have properties train_losses, val_losses.
    # Plot - val_losses not yet properly implemented.
    plt.figure()
    plt.plot(trainer.train_losses[4:], label='Train')
    plt.plot(trainer.val_losses[4:], label='Val')
    plt.ylabel('MSE Loss')
    plt.xlabel('Update step')
    plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
# from jax_md.nn import nequip
#
# # Design a Nequip architecture
# config = nequip.default_config()
# # Spherical harmonic representations
# config.sh_irreps = '1x0e + 1x1e'
# # Hidden representation representations
# config.hidden_irreps = '32x0e + 4x1e'
# config.radial_net_n_hidden = 16
# # Total number of graph net iterations
# config.graph_net_steps = 3
#
# # Dataset shift (average) and scale (std deviation)
# config.shift = -5.73317
# config.scale = 0.030554
#
# config.n_neighbors = 10.
# config.scalar_mlp_std = 4.