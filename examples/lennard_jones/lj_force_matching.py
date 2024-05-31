"""Force matching for Lennard Jones reference data."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str('1,2')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

from pathlib import Path

from jax import random, numpy as jnp
from jax_md import space
import numpy as onp
import optax

from chemtrain import trainers, util, data_processing
from chemtrain.jax_md_mod import custom_space
from util import Initialization

# TODO why slowdown with larger dataset size with multiple gpus??
# user input
save_path = 'output/force_matching/trained_lj.pkl'
used_dataset_size = 2000
subsampling = 1
configuration_str = '../../../../Datasets/LJ/conf_atoms_LJ_10k.npy'
force_str = '../../../../Datasets/LJ/forces_atoms_LJ_10k.npy'
virial_str = '../../../../Datasets/LJ/virial_pressure_LJ_10k.npy'
length_str = '../../../../Datasets/LJ/length_atoms_LJ_10k.npy'
box_side_length = onp.load(length_str)

batch_per_device = 10
batch_cache = 5
epochs = 20
check_freq = 1

box = jnp.ones(3) * jnp.array(box_side_length)

model = 'PairNN'

# load_str only if leading pickled params in init, not for checkpointing!
# --> Delete in publishable version
# load_str = 'saved_models/Energy_params_CGDimeNet_with_ADF.pkl'
load_str = None
checkpoint = None
# checkpoint = 'output/force_matching/Checkpoints/epoch5.pkl'

Path('output/figures').mkdir(parents=True, exist_ok=True)

# build datasets
position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size,
                                            subsampling=subsampling)
force_data = data_processing.get_dataset(force_str, retain=used_dataset_size,
                                         subsampling=subsampling)
virial_data = data_processing.get_dataset(virial_str, retain=used_dataset_size,
                                          subsampling=subsampling)

dataset_size = position_data.shape[0]
print('Dataset size:', dataset_size)

if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.001
elif model == 'PairNN':
    initial_lr = 0.005
else:
    raise NotImplementedError

lr_schedule = optax.exponential_decay(-initial_lr, 1000, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)


position_data = data_processing.scale_dataset_fractional(position_data, box)
R_init = position_data[0]

box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)

model_init_key = random.PRNGKey(0)
# model_init_key, simuation_init_key = random.split(key, 2)

energy_fn_template, _, init_params, nbrs_init = \
    Initialization.select_model(model, R_init, displacement, box,
                                model_init_key, x_vals=None, fractional=True)


trainer = trainers.ForceMatching(init_params, energy_fn_template, nbrs_init,
                                 optimizer, position_data,
                                 force_data=force_data,
                                 batch_per_device=batch_per_device,
                                 batch_cache=batch_cache)

if checkpoint is not None:  # restart from a previous checkpoint
    trainer = util.load_trainer(checkpoint)

trainer.evaluate_mae_testset()
trainer.train(epochs, checkpoint_freq=check_freq, thresh=3)
trainer.evaluate_mae_testset()

trainer.save_trainer(save_path)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(trainer.train_losses, label='Train')
plt.plot(trainer.val_losses, label='Val')
plt.ylabel('MSE Loss')
plt.xlabel('Update step')
plt.savefig('output/figures/force_matching_losses.png')
