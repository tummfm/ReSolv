"""Runs a forward simulation in Jax M.D with loaded parameters.
   Good for trajectory generation for postprocessing and analysis of simulation.
   Can also be used to debug the forward-pass through the simulation.
"""
import os
import sys

from matplotlib.pyplot import sca

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import jax.numpy as jnp
# config.update("jax_debug_nans", True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform, visible_device)

import time

from chemtrain.traj_util import process_printouts, trajectory_generator_init, \
    quantity_traj

import chemtrain.util as util
from chemtrain.jax_md_mod import custom_space
from jax_md import space
import matplotlib.pyplot as plt
import seaborn as sns

import cloudpickle as pickle
from jax.tree_util import tree_map

from util import Postprocessing, Initialization

# import warnings
# warnings.filterwarnings("ignore")
# print("IMPORTANT: You have Warning Messages disabled!")

#TODO:s rename or move into simulation.py
file = 'data/confs/SPC_FW_3nm.gro'  # 905 particles
file = 'data/confs/Water_experimental_3nm.gro'  # 901 particles
# file = 'data/confs/SPC_955_3nm.gro'  # 862 particles
# file = 'data/confs/SPC_FW_2nm.gro'  # 229 particles

# model = 'LJ'
# model = 'Tabulated'
model = 'CGDimeNet'

save_name = 'new_RE_pressure_1ns'
# save_name = '_SPC_1ns'

# saved_trainer_path = '/output/difftre/trained_model.pkl'
# saved_trainer_path = 'output/force_matching/trained_model_8k_20e_testing_new_10.pkl'
# saved_trainer_path = 'output/rel_entropy/trained_model_8k_300up_70ps_095_0005.pkl'
# saved_trainer_path = 'output/force_matching/trained_model_SPC_1k_1e.pkl'
saved_trainer_path = None

saved_params_path = 'output/rel_entropy/trained_params_300up_70ps_8k.pkl'

kbT = 2.49435321
mass = 18.0154
time_step = 0.002  # Bigger time_step possible for CG water?

total_time = 500.01
t_equilib = 0.01
print_every = 0.1

# target_rdf = 'LJ'
# target_rdf = 'SPC'
# target_rdf = 'SPC_FW'
# target_rdf = 'Water_Ox'
target_rdf = 'TIP4P/2005'
rdf_struct = Initialization.select_target_rdf(target_rdf)
# adf_struct = Initialization.select_target_ADF('Water_Ox', 0.318)

# add all target values here, target is only dummy
# target_dict = {'pressure': 1., 'pressure_tensor': 1.}
target_dict = {'rdf': rdf_struct, 'pressure': 1.,'pressure_tensor': 1.}
# target_dict = {'rdf': rdf_struct}

###############################
configuration_str = '../aa_simulations/confs/conf_COM_10k_final.npy'
# configuration_str = '../aa_simulations/confs/conf_SPC.npy'
used_dataset_size = 1000
subsampling  = 1
box_length = jnp.load('../aa_simulations/confs/length_COM_10k_final.npy') #load box length
# box_length = 3.
print('Box length:', box_length)
box = jnp.array([box_length, box_length, 3*box_length])
print('Extended box:',box)

position_data = util.get_dataset(configuration_str, retain=used_dataset_size,
                                 subsampling=subsampling)

# position_data_scale = util.scale_dataset_fractional(position_data, box)
# box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)

# print("box tensor",box_tensor)

# inv_box_tensor = space.inverse(box_tensor)

# print("inv",inv_box_tensor)

R = position_data[0]
R[:,2] += box_length
R = jnp.array(R)

# plt.figure()
# sns.kdeplot(y=scale_fn(R)[:,2].flatten(), label='0 ps')
# plt.savefig('output/surface_tension/intial_scale_R'+save_name+'.png')

# print('max z',jnp.max(R[:,2]))
# print('min z',jnp.min(R[:,2]))
# print('max z',jnp.max(scale_fn(R)[:,2]))
# print('min z',jnp.min(scale_fn(R)[:,2]))
# R = jnp.array(position_data[0])
# R_init = scale_fn(R)
# print(R_init[0])
# print(position_data_scale[0,0])

# print(R[0])
# print(R_init[0])

# box, R, _, _ = io.load_box(file)  # initial configuration

#repulsive prior (sigma, epsilon, r_cut)
constants = {'repulsive': (0.3165, 1., 0.5)}
idxs = {}

simulation_data = Initialization.InitializationClass(
    R_init=R, box=box, kbT=kbT, masses=mass, dt=time_step)
timings = process_printouts(time_step, total_time, t_equilib, print_every)

reference_state, energy_params, simulation_fns, compute_fns, _ = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         target_dict,
                                         wrapped=True,  # bug otherwise
                                         integrator='Nose_Hoover',
                                         prior_idxs=idxs,
                                         prior_constants=constants)

simulator_template, energy_fn_template, neighbor_fn = simulation_fns

if saved_trainer_path is not None:
    print('using loaded trainer')
    loaded_trainer = util.load_trainer(saved_trainer_path)
    # energy_fn_template = loaded_trainer.reference_energy_fn_template #test difference without template
    energy_params = loaded_trainer.params

if saved_params_path is not None:
    print('using saved params')
    with open(saved_params_path, 'rb') as pickle_file:
            params = pickle.load(pickle_file)
            energy_params = tree_map(jnp.array, params)


trajectory_generator = trajectory_generator_init(simulator_template,
                                                 energy_fn_template,
                                                 timings)

# compute trajectory and quantities
t_start = time.time()
traj_state = trajectory_generator(energy_params, reference_state)
print('Generated traj:',traj_state.trajectory.position.shape)
#Save positions of the trajectory
print('First position',traj_state.trajectory.position[0])
jnp.save('output/surface_tension/confs_'+save_name,traj_state.trajectory.position)

print('ps/min: ', total_time / ((time.time() - t_start) / 60.))

assert not traj_state.overflow, ('Neighborlist overflow during trajectory '
                                 'generation. Increase capacity and re-run.')

#long trajectory adjust batch number
quantity_trajectory = quantity_traj(traj_state, compute_fns, energy_params, batch_size=100)

# plotting and prints
from pathlib import Path
Path('output/figures').mkdir(parents=True, exist_ok=True)
if 'rdf' in quantity_trajectory:
    computed_RDF = jnp.mean(quantity_trajectory['rdf'], axis=0)
    Postprocessing.plot_initial_and_predicted_rdf(rdf_struct.rdf_bin_centers,
                                                  computed_RDF, model,
                                                  save_name,
                                                  rdf_struct.reference)

if 'pressure' in quantity_trajectory:
    pressure_traj = quantity_trajectory['pressure']
    print('Size pressure',pressure_traj.shape)
    mean_pressure = jnp.mean(pressure_traj, axis=0)
    std_pressure = jnp.std(pressure_traj, axis=0)
    # we assume samples are iid here: Is approximately true as we only save
    # configurations every ca 100 time steps
    uncertainty_std = jnp.sqrt(jnp.var(pressure_traj) / len(pressure_traj))
    print('Pressure mean:', mean_pressure, 'and standard deviation:',
          std_pressure, 'Statistical uncertaintry STD:', uncertainty_std)
    jnp.save('output/surface_tension/pressure_'+save_name,pressure_traj)

if 'pressure_tensor' in quantity_trajectory:
    pressure_traj = quantity_trajectory['pressure_tensor']
    print('Size tensor pressure',pressure_traj.shape)
    mean_pressure = jnp.mean(pressure_traj, axis=0)
    std_pressure = jnp.std(pressure_traj, axis=0)
    # we assume samples are iid here: Is approximately true as we only save
    # configurations every ca 100 time steps
    uncertainty_std = jnp.sqrt(jnp.var(pressure_traj) / len(pressure_traj))
    print('Pressure scalar mean:', mean_pressure, 'and standard deviation:',
          std_pressure, 'Statistical uncertaintry STD:', uncertainty_std)
    jnp.save('output/surface_tension/pressure_tensor_'+save_name,pressure_traj)
 