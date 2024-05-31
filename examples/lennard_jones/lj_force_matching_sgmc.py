"""Bayesian Machine learning of MD potentials via SG-MCMC."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = str('1,2')
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

from pathlib import Path

import h5py
from jax import device_count, random
from jax_md import space
from jax_sgmc import potential, alias
import matplotlib.pyplot as plt
import numpy as onp
import optax
import time

from chemtrain import trainers, probabilistic, data_processing
from chemtrain.jax_md_mod import custom_space
from util import Initialization


#######################################
# Select sampler and hyperparameters
#######################################
# solver = 'SGLD'
# solver = 'Amagold'
# solver = 'NUTS'
solver = 'Ensemble'
# solver = "SGGMC"


#######################################
# Set important hyper parameters
#######################################
batch_per_device = 5
batch_cache = 2
iterations = 1000
burn_in = 10
lr_reduction = 0.075
num_samples = 20
first_step_size = 0.005
last_step_size = 0.0005
ensemble_size = 3  # how many NN ensembles to use for solver == 'Ensemble'
plot_results = True


batch_size = batch_per_device * device_count()

#######################################
# Setup Data Loader & Import data
#######################################
used_dataset_size = 500
subsampling = 1
save_path = 'output/force_matching/trained_model_param.pkl'

# build datasets

# data = onp.load(data_location_str)
# configuration_str = '/home/student/Datasets/LJ_positions.npy'
# force_str = '/home/student/Datasets/LJ_forces.npy'
# configuration_str = '../../../Datasets/LJ/conf_atoms_LJ_10k.npy'
# force_str = '../../../Datasets/LJ/forces_atoms_LJ_10k.npy'
# virial_str = '../../../Datasets/LJ/virial_pressure_LJ_10k.npy'
# length_str = '../../../Datasets/LJ/length_atoms_LJ_10k.npy'
# box_side_length = onp.load(length_str)
# box = jnp.ones(3) * box_side_length

# position_data = util.get_dataset(configuration_str, retain=used_dataset_size,
#                                  subsampling=subsampling)
# force_data = util.get_dataset(force_str, retain=used_dataset_size,
#                               subsampling=subsampling)
# virial_data = util.get_dataset(virial_str, retain=used_dataset_size,
#                                subsampling=subsampling)

file_location = '../../../../Datasets/LJ/LJ_datasets.h5'
dataset_str = 'Dataset3'
with h5py.File(file_location, 'r') as h5:
    dataset_group = h5.get(dataset_str)
    positions = onp.array(dataset_group.get('positions'))
    forces = onp.array(dataset_group.get('forces'))
    box = onp.array(dataset_group.get('box'))
    temperature = dataset_group.get('temperature')
    density = dataset_group.get('density')

train_ratio = 0.8

Path('output/figures').mkdir(parents=True, exist_ok=True)

positions = positions[:used_dataset_size]
forces = forces[:used_dataset_size]

dataset_size = positions.shape[0]
print('Training dataset size:', dataset_size)

#######################################
# Define experiment and transform data
#######################################
positions = data_processing.scale_dataset_fractional(positions, box)
R_init = positions[0]
box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)

#######################################
# Initialize Model
#######################################
# Select model
# model = 'CGDimeNet'
# model = 'Tabulated'
model = 'PairNN'

if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.001
elif model == 'PairNN':
    initial_lr = 0.005
else:
    raise NotImplementedError

if solver == 'Ensemble':  # needs multiple init_params
    model_init_key = [random.PRNGKey(i) for i in range(ensemble_size)]
else:
    model_init_key = random.PRNGKey(0)

energy_fn_template, neighbor_fn, init_params, nbrs_init = \
    Initialization.select_model(model, R_init, displacement, box,
                                model_init_key, fractional=True)

#######################################
# Build trainer and train
#####################################
if solver == 'Ensemble':
    lr_schedule = optax.exponential_decay(-initial_lr, 1000, lr_reduction)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule)
    )
    trainer_list = []
    for init_params_set in init_params:
        single_trainer = trainers.ForceMatching(
            init_params_set, energy_fn_template, nbrs_init, optimizer,
            positions, force_data=forces,
            batch_per_device=batch_per_device)
        trainer_list.append(single_trainer)
    trainer = trainers.EnsembleOfModels(trainer_list)
    start = time.time()
    trainer.train(iterations, thresh=3)
    end = time.time()
    test_loader = trainer.trainers[0].test_loader
    for sub_trainer in trainer.trainers:
        sub_trainer.evaluate_mae_testset()
else:  # Bayesian learning
    # Define likelihood, prior and combine to MCMC potential
    # TODO use test_loader?
    energy_prior = probabilistic.uniform_prior
    (prior, likelihood, init_samples, train_loader, val_loader, test_loader
     ) = probabilistic.init_force_matching(
            energy_prior, energy_fn_template, nbrs_init, init_params,
            positions, force_data=forces, force_scale=200.
    )
    if solver == 'NUTS':
        warmup_steps = 1000
        batch_cache = 1
        trainer = trainers.NUTSForceMatching(
            prior, likelihood, train_loader, init_samples[0], batch_cache,
            batch_size, val_loader, warmup_steps
        )

    else:  # SG-MCMC sampling
        # select parallelization strategy for mini-batched potentials
        potential_fn = potential.minibatch_potential(prior, likelihood,
                                                     strategy='vmap')
        full_potential_fn = potential.full_potential(prior, likelihood,
                                                     strategy='vmap')
        if solver == 'Amagold':
            sgmcmc_solver = alias.amagold(
                potential_fn, full_potential_fn, train_loader,
                cache_size=batch_size*batch_cache, batch_size=batch_size,
                first_step_size=initial_lr,
                last_step_size=initial_lr*lr_reduction, burn_in=burn_in
            )
        elif solver == 'SGLD':
            sgmcmc_solver = alias.sgld(
                potential_fn, train_loader, cache_size=batch_size*batch_cache,
                batch_size=batch_size, first_step_size=initial_lr,
                last_step_size=initial_lr*lr_reduction, burn_in=burn_in,
                rms_prop=True, accepted_samples=num_samples
            )
        elif solver == 'SGGMC':
            sgmcmc_solver = alias.sggmc(
                potential_fn, full_potential_fn, train_loader,
                cache_size=batch_size*batch_cache, batch_size=batch_size,
                first_step_size=first_step_size, last_step_size=last_step_size,
                burn_in=burn_in
                )
        else:
            raise ValueError(f'Solver {solver} not recognized.')

        trainer = trainers.SGMCForceMatching(
            sgmcmc_solver, init_samples, None,
            energy_fn_template=energy_fn_template
        )

    start = time.time()
    trainer.train(iterations)
    end = time.time()

maes = probabilistic.validation_mae_params_fm(
    trainer.list_of_params, test_loader, energy_fn_template, nbrs_init,
    box_tensor, batch_size, batch_cache)

#######################################
# Post Processing / Saving
#######################################

# Print out elapsed time (per iteration)
print(f'Training finished: Average time:'
      f' {onp.round((end - start) / iterations, 4)} s  per iteration')

trainer.save_energy_params(file_path=save_path, save_format='.pkl')
results = trainer.results

if plot_results:
    # TODO: Identifier for ensemble will need different inputs!
    # Create unique identifier --> important for hyper-parameter studies
    initial_lr_label = str(initial_lr)[str(initial_lr).find('.')+1:]
    lr_reduction_label = str(lr_reduction)[str(lr_reduction).find('.')+1:]
    identifier = [solver + '_IT' + str(iterations) + 'B' + str(burn_in) +
                  'ILR' + initial_lr_label + 'LR' + lr_reduction_label][0]

    if solver in ['NUTS', 'Ensemble']:
        pass
    else:
        # STD
        plt.figure()
        plt.plot(results[0]['samples']['variables']['F_std'])
        plt.ylabel('STD')
        plt.xlabel('Iterations (?)')
        plt.savefig(['/home/student/Gregor/myjaxmd/output/figures/'
                     + identifier + '.png'][0])
