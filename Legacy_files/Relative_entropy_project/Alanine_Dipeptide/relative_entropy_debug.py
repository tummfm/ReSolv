import os
import sys

import chemtrain.util
from chemtrain.jax_md_mod.custom_quantity import dihedral_displacement

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform, visible_device)

import optax

import jax.numpy as jnp
import numpy as onp
from jax import random, vmap, lax
from jax.tree_util import tree_map
from chemtrain import util, trainers, traj_util
from chemtrain.jax_md_mod import io, custom_space
from jax_md import space
from util import Initialization, visualization
import matplotlib.pyplot as plt
from aa_simulations.Legacy_files import evaluate_fn
import aa_simulations.aa_util as aa
from functools import partial
import time
import cloudpickle as pickle

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")


# user input
mapping = 'heavy'
#TODO: combine these input files?
# name = 'RE_50up_1ns_pretrain_bonds_400k'
name = 'RE_150up_40k'
folder_name = f'models_RE_nolj_40x10ns/{name}/'
folder_training_name = f'models_RE_nolj_40x10ns/{name}/training/'
labels = ['Reference',name,'RE 400k 3C 1ns']
save_epochs = [0,1,2,3,4,5,10,15,20,30,40,50,100]

file_topology = f'data/confs/Alanine_dipeptide_{mapping}_2_7nm.gro'
save_path = f'output/rel_entropy/trained_params_alanine_{mapping}_{name}.pkl'
save_checkpoint = f'output/rel_entropy/{name}/checkpoint_model_alanine_{mapping}_{name}'
# save_checkpoint_points = f'output/rel_entrpy/checkpoint_points_alanine_{mapping}_{name}.npy'
save_plot = f'output/figures/RE_gradient_norm_alanine_{mapping}_{name}.png'
used_dataset_size = 400000
subsampling = 1
n_trajectory = 10
simulate = False
n_forward = 40 # number of forward trajectories

configuration_str = f'../aa_simulations/alanine_dipeptide/confs/confs_{mapping}_100ns.npy'
force_str = f'../aa_simulations/alanine_dipeptide/confs/forces_{mapping}_100ns.npy'

# saved_params_path = 'output/force_matching/trained_params_alanine_heavy_FM_50epochs_440k_500batchs_lr001_newFM.pkl'
saved_params_path = None

# simulation parameters
system_temperature = 300  # Kelvin
Boltzmann_constant = 0.0083145107  # in kJ / mol K
kbT = system_temperature * Boltzmann_constant

time_step = 0.002 #2f -> 4f
total_time = 1005 #-> increase 200
t_equilib = 5.
print_every = 0.2 #->0.1ps

time_step_forward = 0.002  # For SPC/FW 1fs time step necessary
total_time_forward = 11000
t_equilib_forward = 11000.
print_every_forward = 0.2

model = 'CGDimeNet'
# model = 'Tabulated'

# checkpoint = 'output/rel_entropy/Checkpoints/epoch2.pkl'
checkpoint = None
check_freq = None

num_updates = 150
if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.003 #-> 0.001
else:
    raise NotImplementedError

lr_schedule = optax.exponential_decay(-initial_lr, num_updates, 0.1) #->300, 0.01
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

box, R, masses, species, bonds = io.load_box(file_topology) # initial configuration

print(species)
print('distinguish C atoms')
# species = jnp.array([6, 1, 8, 7, 6, 6, 1, 8, 7, 6])
species = jnp.array([6, 1, 8, 7, 2, 6, 1, 8, 7, 6])
print(species)

box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
displacement, shift = space.periodic_general(box_tensor,
                                    fractional_coordinates=True, wrapped=True)

# prior_dict = {'bond': 1., 'angle': 1., 'LJ': 1.}
# prior_dict = {'bond': 1., 'angle': 1., 'LJ': 1., 'dihedral': 1.}
prior_dict = {'bond': 1., 'angle': 1., 'dihedral': 1.}
# prior_dict = {'bond': 1., 'angle': 1.}
# prior_dict = {'bond': 1.}
# prior_dict = {}
idxs, constants = Initialization.select_protein('heavy_alanine_dipeptide', prior_dict)

# _, test_data = util.get_dataset(configuration_str,
#             retain=200000, subsampling=subsampling,test_split=True)

position_data = util.get_dataset(configuration_str, retain=used_dataset_size,
                                 subsampling=subsampling)

# force_data = util.get_dataset(force_str, retain=used_dataset_size,
#                                  subsampling=subsampling)
# _, test_forces = util.get_dataset(force_str,
#             retain=200000, subsampling=subsampling, test_split=True)

#Random starting points
key = random.PRNGKey(0)
R_init = random.choice(key,position_data,(1,n_trajectory),replace=False)
# # random choice prevents use of int as shape (create issue?), numpy version allows this
R_init = R_init.reshape(-1,10,3)

# R_init = jnp.array([position_data[29264],position_data[49712],position_data[178002],
# position_data[28894],position_data[107817],position_data[93472],
# position_data[21073],position_data[111660],position_data[1731],position_data[33123]])


simulation_data = Initialization.InitializationClass(R_init=R_init,
        box=box, kbT=kbT, masses=masses, dt=time_step, species=species)

t_start_init = time.time()
reference_state, init_params, simulation_fns, _, _ = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         integrator='Langevin',
                                         prior_constants=constants,
                                         prior_idxs=idxs)
t_end_init = time.time() - t_start_init
print(f'total time for init:', t_end_init/ 60,'mins')
print('initialization end')

print('dimension sim state', reference_state[0].position.ndim)
if reference_state[0].position.ndim > 2:
    n_traj = reference_state[0].position.shape[0]
    print('Number of trajectories', n_traj)
else:
    n_traj = 1
    print('Number of trajectories', n_traj)
simulator_template, energy_fn_template, neighbor_fn = simulation_fns

reference_data = util.scale_dataset_fractional(position_data, box)
print('reference_data',reference_data.shape)

if saved_params_path is not None:
    print('using saved params')
    with open(saved_params_path, 'rb') as pickle_file:
            params = pickle.load(pickle_file)
            init_params = tree_map(jnp.array, params)

trainer = trainers.RelativeEntropy(init_params, optimizer, reweight_ratio=1.1,
                                        energy_fn_template=energy_fn_template)
#change folder checkpoint checkpoint_folder='Checkpoints
trainer.add_statepoint(reference_data, energy_fn_template, simulator_template,
                                    neighbor_fn, timings, kbT, reference_state,
                                    # vmap_batch=1)
                                    reference_batch_size=used_dataset_size, vmap_batch=10000)

# learning
if checkpoint is not None:  # restart from a previous checkpoint
    trainer = util.load_trainer(checkpoint)

# prediction_fn_template = evaluate_fn.init_single_prediction(util.tree_get_single(
#                                         reference_state[1]), energy_fn_template)

batch_size = 1000
# test_positions = util.scale_dataset_fractional(test_data, box)
# test_positions = test_positions[:-1] #100k test set instead of 1000001
# # test_forces = test_forces[:-1]
# batched_positions = test_positions.reshape((-1,batch_size,10,3))

psi_indices, phi_indices = [3, 4, 6, 8], [1, 3, 4, 6]
phi_angle_ref = vmap(aa.one_dihedral_displacement, (0,None,None))(reference_data,
                                                      displacement, phi_indices)
psi_angle_ref = vmap(aa.one_dihedral_displacement, (0,None,None))(reference_data,
                                                      displacement, psi_indices)
dihedral_angles_ref = jnp.stack((phi_angle_ref,psi_angle_ref),axis=1)
h_ref, _, _ = jnp.histogram2d(dihedral_angles_ref[:,0],dihedral_angles_ref[:,1],
                                                        bins = 60, density=True)

# mse = onp.zeros(num_updates)
# mae = onp.zeros(num_updates)
kl_div = onp.zeros(num_updates)
mse_energy = onp.zeros(num_updates)
js_div = onp.zeros(num_updates)


# print(util.scale_dataset_fractional(R_init,box)[0])
# print('ref state')
# print(reference_state[0].position[0])
# print('init point')
# print(trainer.init_points[0][0].position[0])


# after_0 = trainer.get_sim_state(0)
# print('epoch 0')
# print(after_0[0].position[0])
# print(trainer.trajectory_states[0].sim_state[0].position[0])
# print(trainer.init_points[0][0].position[0])

# trainer.train(1, checkpoint_freq=check_freq)
# after_1 = trainer.get_sim_state(0)
# print('epoch 1')
# print(after_1[0].position[0])
# print(trainer.trajectory_states[0].sim_state[0].position[0])
# print(trainer.init_points[0][0].position[0])

# trainer.train(1, checkpoint_freq=check_freq)
# after_2 = trainer.get_sim_state(0)
# print('epoch 2')
# print(after_2[0].position[0])
# print(trainer.trajectory_states[0].sim_state[0].position[0])
# print(trainer.init_points[0][0].position[0])

# trainer.train(1, checkpoint_freq=check_freq)
# after_3 = trainer.get_sim_state(0)
# print('epoch 3')
# print(after_3[0].position[0])
# print(trainer.trajectory_states[0].sim_state[0].position[0])
# print(trainer.init_points[0][0].position[0])

from pathlib import Path
Path(f'output/rel_entropy/{name}').mkdir(parents=True, exist_ok=True)
Path(f'plots/postprocessing/{folder_name}').mkdir(parents=True, exist_ok=True)
Path(f'plots/postprocessing/{folder_training_name}').mkdir(parents=True, exist_ok=True)
run_once = 0 #maybe find better way to do this? Integrate in trainer?
for k in range(num_updates):
    if k < 500:
        print(f'Training with {total_time-t_equilib} ps')
        trainer.train(1, checkpoint_freq=check_freq)
        if k in save_epochs:
            trainer.save_energy_params(save_checkpoint+f'_epoch{k}.pkl','.pkl')
    # elif k < 150:
    #     print('Training with 1ns')
    #     if run_once == 0:
    #         trainer.save_energy_params(save_checkpoint+f'_epoch{k}.pkl','.pkl')
    #         print('saved e params')
    #         # jnp.save(save_checkpoint_points, trainer.trajectory_states[0].sim_state[0].position)
    #         # print('saved init points')
    #         timings = traj_util.process_printouts(time_step, 1005, 5.,
    #                                     print_every)
    #         ref_states = trainer.trajectory_states[0].sim_state
    #         trainer.add_statepoint(reference_data, energy_fn_template, simulator_template,
    #                             neighbor_fn, timings, kbT, ref_states, reference_batch_size=40000,
    #                                               set_key=0, initialize_traj=True, vmap_batch=1000)
    #         run_once = 1
    #     trainer.train(1, checkpoint_freq=check_freq)
    # # else:
    # #     print('Training with 400ps')
    #     if run_once == 1:
    #         trainer.save_energy_params(save_checkpoint+f'_epoch{k}.pkl','.pkl')
    #         print('saved e params')
    #         # jnp.save(save_checkpoint_points, trainer.trajectory_states[0].sim_state[0].position)
    #         # print('saved init points')
    #         timings = traj_util.process_printouts(time_step, 410, 10.,
    #                                     print_every)
    #         ref_states = trainer.trajectory_states[0].sim_state
    #         trainer.add_statepoint(reference_data, energy_fn_template, simulator_template,
    #                             neighbor_fn, timings, kbT, ref_states, reference_batch_size=40000,
    #                                               set_key=0, initialize_traj=True, vmap_batch=100)
    #         run_once = 2
    #     trainer.train(1, checkpoint_freq=check_freq)
    # else:
    #     print('Training with 400ps')
    #     if run_once == 2:
    #         timings = traj_util.process_printouts(time_step, 400, t_equilib,
    #                                     print_every)
    #         trainer.add_statepoint(reference_data, energy_fn_template, simulator_template,
    #                             neighbor_fn, timings, kbT, reference_state, reference_batch_size=40000,
    #                                                                     set_key=0, initialize_traj=True)
    #         run_once = 3
    #     trainer.train(1, checkpoint_freq=check_freq)

    t_start = time.time()
    # prediction_fn = partial(prediction_fn_template, trainer.params)
    # batched_predictions = lax.map(vmap(prediction_fn), batched_positions)
    # predictions = batched_predictions['F'].reshape((-1,10,3))
    # print('predictions',predictions.shape)
    # print(jnp.mean(predictions))
    # print(jnp.mean(test_forces))

    # mse[i] = util.mse_loss(predictions, test_forces)
    # mae[i] = util.mae_loss(predictions, test_forces)

    phi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(
                            trainer.trajectory_states[0].trajectory.position,
                                                    displacement, phi_indices)
    psi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(
                            trainer.trajectory_states[0].trajectory.position,
                                                    displacement, psi_indices)
    dihedral_angles = jnp.stack((phi_angle,psi_angle),axis=1)
    
    visualization.plot_histogram_free_energy(dihedral_angles,name+f'_epoch{k}',folder=folder_training_name)

    h, _, _ = jnp.histogram2d(dihedral_angles[:,0],dihedral_angles[:,1],
                                                     bins = 60, density=True)

    kl_div[k] = evaluate_fn.kl_nonzero(h_ref,h)
    mse_energy[k] = aa.MSE_energy(h_ref,h,kbT)
    js_div[k] = chemtrain.util.jenson_shannon(h_ref, h)
    t_end = time.time() - t_start
    print(f'total time properties Epoch {k}:', t_end/ 60,'mins')

#make loop and calculate angles
#mke loop with increasing size
print('training done')
# trainer.save_trainer(save_path) # save parameters
#get time here for saving?
trainer.save_energy_params(save_path,'.pkl')
print('saved e params')
# trainer.save_init_points(0, save_init_path, save_format='.npy')
# print('saved init points')

Path(f'output/figures/losses/{name}').mkdir(parents=True, exist_ok=True)
# plt.figure()
# plt.plot(mse)
# plt.savefig(f'output/figures/losses/{name}/RE_mse_alanine_{mapping}_{name}.png')

# plt.figure()
# plt.plot(mae)
# plt.savefig(f'output/figures/losses/{name}/RE_mae_alanine_{mapping}_{name}.png')

plt.figure()
plt.plot(kl_div)
plt.savefig(f'output/figures/losses/{name}/RE_KLS_alanine_{mapping}_{name}.png')

plt.figure()
plt.plot(mse_energy)
plt.savefig(f'output/figures/losses/{name}/RE_MSE_energy_alanine_{mapping}_{name}.png')

plt.figure()
plt.plot(js_div)
plt.savefig(f'output/figures/losses/{name}/RE_JS_alanine_{mapping}_{name}.png')

print('plot done')

# test retrieving params and energy function
# energy_fn = trainer.energy_fn
# trainer_loaded = util.load_trainer(save_path)
# energy_fn_2 = energy_fn_template(trainer_loaded.params)
# print('Precited energies:', energy_fn(reference_state[0].position,
# reference_state[1]), energy_fn_2(reference_state[0].position,
# reference_state[1]))

plt.figure()
plt.plot(trainer.gradient_norm_history)
plt.yscale('log')
plt.savefig(save_plot)

if simulate:
    print('forward simulation')
    energy_params = trainer.params

    R = position_data[0]
    R_init = onp.full((n_forward,10,3), R)
    print(R_init.shape)

    simulation_data_forward = Initialization.InitializationClass(
    R_init=R_init, box=box, kbT=kbT, masses=masses, dt=time_step, species=species)
    timings = traj_util.process_printouts(time_step_forward, total_time_forward, t_equilib_forward, print_every_forward)

    reference_state, _, simulation_fns, _, _ = \
    Initialization.initialize_simulation(simulation_data_forward,
                                         model,
                                         integrator='Langevin',
                                         prior_constants=constants,
                                         prior_idxs=idxs)

    simulator_template, _, _ = simulation_fns

    trajectory_generator = traj_util.trajectory_generator_init(simulator_template,
                                                    energy_fn_template,
                                                    timings)
    # compute trajectory and quantities
    t_start = time.time()
    traj_state = trajectory_generator(energy_params,reference_state) #add kbt to debug
    t_end = time.time() - t_start
    print('total time:', t_end)
    print('number of parallel trajectories', n_forward)
    print('ns/h: ', n_forward*(total_time/1000 / (t_end / 3600.)))

    jnp.save('output/alanine_confs/confs_alanine_'+
                                        name,traj_state.trajectory.position)
    print(traj_state.trajectory.position.shape)
    assert not traj_state.overflow, ('Neighborlist overflow during trajectory '
                                    'generation. Increase capacity and re-run.')

    print(time_step)
    temp = traj_state.aux['kbT']
    mean_temp = jnp.mean(temp)
    print('average kbT:',mean_temp,'vs reference:',kbT)
    # visualization.plot_temp(traj_state,kbT,plotname)

    positions = traj_state.trajectory.position
    positions_compare = onp.load('output/alanine_confs/'+
                    'confs_alanine_RE_1ns_400k_3C.npy')
    
    #dihedral
    psi_indices, phi_indices = [3, 4, 6, 8], [1, 3, 4, 6]

    phi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(positions,
                                                        displacement, phi_indices)
    psi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(positions,
                                                        displacement, psi_indices)
    phi_angle_compare = vmap(aa.one_dihedral_displacement, (0,None,None))(positions_compare,
                                                        displacement, phi_indices)
    psi_angle_compare = vmap(aa.one_dihedral_displacement, (0,None,None))(positions_compare,
                                                        displacement, psi_indices)

    phi_angle_ref = vmap(aa.one_dihedral_displacement, (0,None,None))(
                                        reference_data, displacement, phi_indices)
    psi_angle_ref = vmap(aa.one_dihedral_displacement, (0,None,None))(
                                        reference_data, displacement, psi_indices)

    dihedral_angles = jnp.stack((phi_angle,psi_angle),axis=1)
    print(dihedral_angles.shape)
    dihedral_angles_compare = jnp.stack((phi_angle_compare,psi_angle_compare),axis=1)
    print(dihedral_angles_compare.shape)
    dihedral_ref = jnp.stack((phi_angle_ref,psi_angle_ref),axis=1)

    #Random seed atomistic reference simulations
    phi_angles_random = onp.load('../aa_simulations/alanine_dipeptide/confs/phi_angles_r100ns.npy')
    psi_angles_random = onp.load('../aa_simulations/alanine_dipeptide/confs/psi_angles_r100ns.npy')

    phi = phi_angle.reshape((n_forward,-1))
    psi = psi_angle.reshape((n_forward,-1))
    dihedral_angles_split = dihedral_angles.reshape((n_forward,-1,2))
    print(dihedral_angles_split.shape)

    phi_compare = phi_angle_compare.reshape((n_forward,-1))
    psi_compare = psi_angle_compare.reshape((n_forward,-1))
    dihedral_angles_compare_split = dihedral_angles_compare.reshape((n_forward,-1,2))

    for i in range(n_forward):
        visualization.plot_histogram_free_energy(dihedral_angles_split[i,:],
                        name+f'_{i}',folder=folder_name)
    
    visualization.plot_1d_dihedral([phi_angles_random, phi, phi_compare], 'phi_' + name,
                                   labels=labels, folder=folder_name)
    visualization.plot_1d_dihedral([psi_angles_random, psi, psi_compare], 'psi_' + name,
                                   labels=labels, xlabel='$\psi$', folder=folder_name)

    visualization.plot_compare_1d_free_energy([phi_angle, phi_angle_compare], dihedral_ref[:, 0],
                                                    'phi_' + name,labels[1:], folder=folder_name)
    visualization.plot_compare_1d_free_energy([psi_angle, psi_angle_compare], dihedral_ref[:, 1],
                                                    'psi_' + name,labels[1:], folder=folder_name,
                                              xlabel='$\psi$')
    visualization.plot_histogram_free_energy(dihedral_angles,name+'_00all',folder=folder_name)

    ################################################################################
    # Criteria
    dihedral_ref = jnp.deg2rad(dihedral_ref)
    dihedral_angles_split = jnp.deg2rad(dihedral_angles_split) #convert to rad
    dihedral_angles_compare_split = jnp.deg2rad(dihedral_angles_compare_split)
    nbins = 60

    h_ref, y1, y2  = onp.histogram2d(dihedral_ref[:,0],dihedral_ref[:,1], bins=nbins, density=True)

    mse = []
    mse_compare = []

    for j in range(n_forward):
        h, _, _  = onp.histogram2d(dihedral_angles_split[j,:,0],dihedral_angles_split[j,:,1],
                                                                    bins=nbins, density=True)
        h_compare, _, _  = onp.histogram2d(dihedral_angles_compare_split[j,:,0],
                                    dihedral_angles_compare_split[j,:,1], bins=nbins, density=True)
        mse.append(aa.MSE_energy(h_ref,h,kbT))
        mse_compare.append(aa.MSE_energy(h_ref,h_compare,kbT))

    print(name)
    print('mse, (mean/std)')
    print(onp.mean(mse),onp.std(mse))
    print(onp.min(mse),onp.max(mse))
    print('compare')
    print('mse, (mean/std)')
    print(onp.mean(mse_compare),onp.std(mse_compare))
    print(onp.min(mse_compare),onp.max(mse_compare))

    dihedral_angles = jnp.deg2rad(dihedral_angles)
    dihedral_angles_compare = jnp.deg2rad(dihedral_angles_compare)

    h_all, _, _  = onp.histogram2d(dihedral_angles[:,0],dihedral_angles[:,1], bins=nbins, density=True)
    h_all_compare, _, _  = onp.histogram2d(dihedral_angles_compare[:,0],dihedral_angles_compare[:,1],
                                                                            bins=nbins, density=True)
    print('all mse')
    mse_all = aa.MSE_energy(h_ref,h_all,kbT)
    mse_all_compare = aa.MSE_energy(h_ref,h_all_compare,kbT)
    print(mse_all)
    print('all compare mse')
    print(mse_all_compare)

