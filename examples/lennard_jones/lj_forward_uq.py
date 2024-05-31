"""Runs a forward simulation in Jax M.D with loaded parameters.
   Good for trajectory generation for postprocessing and analysis of simulation.
   Can also be used to debug the forward-pass through the simulation.
"""

import os
from pathlib import Path
import time

visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import cloudpickle as pickle
from jax import lax, numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

from chemtrain import traj_util, probabilistic, util, dropout
from util import Postprocessing, Initialization


mode = ('output_uq')
samples = 5  # TODO: Currently not used by infer_output_uncertainty (all sample sprovided are analysed) // Change this

model = 'PairNN'  # 'PairNN', 'CGDimeNet', 'LJ', 'Tabulated'

saved_energy_param = '/home/student/Gregor/myjaxmd/output/force_matching/trained_model_param.pkl'

system_temperature = 296.15  # Kelvin
Boltzmann_constant = 0.0083145107  # in kJ / mol K
kbT = system_temperature * Boltzmann_constant
mass = 18.0154
time_step = 0.002  # Bigger time_step possible for CG water?

total_time = 30.
t_equilib = 5.
print_every = 0.1

# target_rdf = 'LJ' # 'LJ', 'SPC', 'SPC_FW', 'Water_Ox'
rdf_struct = Initialization.select_target_rdf('Water_Ox')
# adf_struct = Initialization.select_target_ADF('Water_Ox', 0.318)

# add all target values here, target is only dummy
target_dict = {'rdf': rdf_struct}
# target_dict = {'rdf': rdf_struct, 'adf': adf_struct, 'pressure': 1.}
# target_dict = {'rdf': rdf_struct, 'pressure_scalar': 1.}
# target_dict = {'rdf': rdf_struct, 'pressure': 1.}
# target_dict = {'pressure': 1.}


##### Load R and box as it is done in force_sgmc
from chemtrain.data_processing import get_dataset

configuration_str =  '/home/student/Datasets/LAMMPS/conf_atoms_LJ_10k.npy'
length_str =  '/home/student/Datasets/LAMMPS/length_atoms_LJ_10k.npy'
box_side_length = onp.load(length_str)
box = jnp.ones(3) * box_side_length
used_dataset_size = 1000
subsampling = 1
position_data = get_dataset(configuration_str, retain=used_dataset_size,
                            subsampling=subsampling)
R = position_data[0] 

###########################################################

simulation_data = Initialization.InitializationClass(
    R_init=R, box=box, kbT=kbT, masses=mass, dt=time_step)
timings = traj_util.process_printouts(time_step, total_time,
                                      t_equilib, print_every)

reference_state, energy_params, simulation_fns, quantities, _ = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         target_dict=target_dict,
                                         wrapped=True,  # bug otherwise
                                         integrator='Nose_Hoover')

simulator_template, energy_fn_template, neighbor_fn = simulation_fns

if saved_energy_param is not None:
    with open(saved_energy_param, 'rb') as f:
        energy_params = pickle.load(f)


energy_params = util.tree_get_slice(energy_params, 0, samples)

trajectory_generator = traj_util.trajectory_generator_init(simulator_template,
                                                           energy_fn_template,
                                                           timings)

start = time.time()
inference_temperatures = [296.15, 325, 350, 400]

if 'deterministic' in mode:
    if dropout.dropout_is_used(energy_params):
        # in deterministic case, we simply run the model with dropout disabled
        deterministic_params, _ = dropout.split_dropout_params(energy_params)
    else:
        deterministic_params = energy_params

    # compute trajectory and quantities
    t_start = time.time()
    traj_state = trajectory_generator(deterministic_params, reference_state)
    print('ps/min: ', total_time / ((time.time() - t_start) / 60.))

    quantity_traj = traj_util.quantity_traj(traj_state, quantities,
                                            deterministic_params)

    # plotting and prints
    # TODO move into dedicated Postprocessing - avoid code duplication below
    Path('output/figures').mkdir(parents=True, exist_ok=True)
    if 'rdf' in quantity_traj:
        computed_RDF = jnp.mean(quantity_traj['rdf'], axis=0)
        Postprocessing.plot_initial_and_predicted_rdf(rdf_struct.rdf_bin_centers,
                                                      computed_RDF, model,
                                                      visible_device,
                                                      rdf_struct.reference_rdf)

    if 'pressure' in quantity_traj:
        pressure_traj = quantity_traj['pressure']
        mean_pressure = jnp.mean(pressure_traj, axis=0)
        std_pressure = jnp.std(pressure_traj, axis=0)
        # we assume samples are iid here: Is approximately true as we only save
        # configurations every ca 100 time steps
        uncertainty_std = jnp.sqrt(jnp.var(pressure_traj) / len(pressure_traj))
        print('Pressure mean:', mean_pressure, 'and standard deviation:',
              std_pressure, 'Statistical uncertaintry STD:', uncertainty_std)

    if 'pressure_scalar' in quantity_traj:
        pressure_traj = quantity_traj['pressure_scalar']
        mean_pressure = jnp.mean(pressure_traj, axis=0)
        std_pressure = jnp.std(pressure_traj, axis=0)
        # we assume samples are iid here: Is approximately true as we only save
        # configurations every ca 100 time steps
        uncertainty_std = jnp.sqrt(jnp.var(pressure_traj) / len(pressure_traj))
        print('Pressure scalar mean:', mean_pressure, 'and standard deviation:',
              std_pressure, 'Statistical uncertaintry STD:', uncertainty_std)

    if 'adf' in quantity_traj:
        computed_ADF = jnp.mean(quantity_traj['adf'], axis=0)
        Postprocessing.plot_initial_and_predicted_adf(adf_struct.adf_bin_centers,
                                                      computed_ADF, model,
                                                      visible_device,
                                                      adf_struct.reference_adf)

if 'snapshot_uq' in mode:
    assert dropout.dropout_is_used(energy_params), ('Currently only prediction'
                                                    ' with dropout UQ'
                                                    ' implemented')
    force_std = probabilistic.init_force_uq(
        energy_fn_template, n_splits=samples, vmap_batch_size=2)
    std_tracectories = []
    for T in inference_temperatures:
        kt_schedule = lambda t: T * Boltzmann_constant
        full_params, _ = dropout.split_dropout_params(energy_params)
        traj_state = trajectory_generator(full_params, reference_state,
                                          kt_schedule=kt_schedule)

        def force_std_traj(state):
            nbrs = neighbor_fn(state.position, reference_state[1])
            return force_std(energy_params, (state, nbrs))

        std_trajectory = lax.map(force_std_traj, traj_state.trajectory)
        std_tracectories.append(std_trajectory)
        print(std_trajectory)

    plt.Figure()
    for i, trajectory in enumerate(std_tracectories):
        plt.plot(trajectory, label=str(inference_temperatures[i]))
    plt.savefig('output/figures/force_uncertainty.png')


if 'output_uq' in mode:

    for T in inference_temperatures:

        kt_schedule = lambda t: T * Boltzmann_constant
        predictions = probabilistic.infer_output_uncertainty(
            energy_params, reference_state, trajectory_generator, quantities,
            samples, kt_schedule=kt_schedule, vmap_simulations_per_device=1)
        statistics = probabilistic.mcmc_statistics(predictions)

        Path('output/figures').mkdir(parents=True, exist_ok=True)
        if 'rdf' in statistics:
            Postprocessing.plot_initial_and_predicted_rdf(
                rdf_struct.rdf_bin_centers, statistics['rdf']['mean'], model,
                visible_device, rdf_struct.reference)

        """
        if 'pressure' in statistics:
            print('Pressure ensemble mean:', statistics['pressure']['mean'],
                  'and standard deviation:', statistics['pressure']['std'])

        if 'adf' in statistics:
            Postprocessing.plot_initial_and_predicted_adf(
                adf_struct.adf_bin_centers, statistics['adf']['mean'], model,
                visible_device, adf_struct.reference_adf,
                std=statistics['adf']['std'], T=T)
        """
        
end = time.time()
print("Simulated " + str(samples) + " samples in " + str(end - start) + 
" seconds -- " + str(onp.round((end - start)/ samples, 4)) + "s/iteration")
