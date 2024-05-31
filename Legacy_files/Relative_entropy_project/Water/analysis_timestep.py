'''Runs a forward simulation in Jax M.D with loaded parameters.
   Good for trajectory generation for postprocessing and analysis of simulation.
   Can also be used to debug the forward-pass through the simulation.
'''
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import jax.numpy as jnp
# config.update('jax_debug_nans', True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform, visible_device)

import time

from chemtrain.traj_util import process_printouts, trajectory_generator_init, \
    quantity_traj, custom_quantity
from chemtrain.jax_md_mod import io
import chemtrain.util as c_util
from util import Initialization
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")

file = 'data/confs/SPC_FW_3nm.gro'  # 905 particles
file = 'data/confs/Water_experimental_3nm.gro'  # 901 particles
# file = 'data/confs/SPC_955_3nm.gro'  # 862 particles
# file = 'data/confs/SPC_FW_2nm.gro'  # 229 particles

# model = 'LJ'
# model = 'Tabulated'
model = 'CGDimeNet'

# plotname = 'FM_8k_1e_vt_60ps'
plotname = 'FM_energies_200ps'

# saved_trainer_path = '../notebooks/saved_models/CG_water_GNN.pkl'
# saved_trainer_path = 'output/difftre/trained_model.pkl'
# saved_trainer_path = 'output/force_matching/trained_model_8k_1e_prior.pkl'
# saved_trainer_path = 'output/rel_entropy/trained_model_8k_297up.pkl'
# saved_trainer_path = 'output/rel_entropy/trained_model_8k_300up_70ps_095.pkl'
saved_trainer_path = None

kbT = 2.49435321
mass = 18.0154
time_step = 0.01  # Bigger time_step possible for CG water?

total_time = 200
t_equilib = 0.1
print_every = 0.1

# target_rdf = 'LJ'
# target_rdf = 'SPC'
# target_rdf = 'SPC_FW'
# target_rdf = 'Water_Ox'
target_rdf = 'TIP4P/2005'
# rdf_struct = Initialization.select_target_RDF(target_rdf)
# tcf_struct = Initialization.select_target_TCF('TIP4P/2005', 0.8, nbins=30)

# adf_struct = Initialization.select_target_ADF('TIP4P/2005', 0.318)

# add all target values here, target is only dummy
# target_dict = {'rdf': rdf_struct, 'adf': adf_struct, 'pressure': 1.}
# target_dict = {'rdf': rdf_struct, 'pressure_scalar': 1.}
# target_dict = {'rdf': rdf_struct, 'pressure': 1.}
# target_dict = {'pressure': 1.}
# target_dict = {'rdf': rdf_struct}
# target_dict = {'adf': adf_struct}
# target_dict = {'tcf': tcf_struct}
target_dict = None


###############################
# box, R, _, _ = io.load_box(file)  # initial configuration

configuration_str = '../aa_simulations/confs/conf_COM_10k_final.npy'
used_dataset_size = 1000
subsampling  = 1
box_length = jnp.load('../aa_simulations/confs/length_COM_10k_final.npy') #load box length
# box_length = 3.
print('Box length:', box_length)
box = jnp.ones(3) * box_length

position_data = c_util.get_dataset(configuration_str, retain=used_dataset_size,
                                subsampling=subsampling)
R = jnp.array(position_data[0])

time_array = [0.002, 0.005, 0.01]
# time_array = [0.01, 0.008, 0.005, 0.002]
energies = []
kinetic_energies = []
total_energies = []

from pathlib import Path
Path(f'output/energy/{plotname}').mkdir(parents=True, exist_ok=True)

for dt in time_array:
    simulation_data = Initialization.InitializationClass(
        R_init=R, box=box, kbT=kbT, masses=mass, dt=dt)
    timings = process_printouts(dt, total_time, t_equilib, print_every)


    reference_state, energy_params, simulation_fns, compute_fns, _ = \
        Initialization.initialize_simulation(simulation_data,
                                            model,
                                            target_dict,
                                            wrapped=True,  # bug otherwise
                                            #  integrator='Nose_Hoover')
                                            integrator='NVE')

    simulator_template, energy_fn_template, neighbor_fn = simulation_fns

    if saved_trainer_path is not None:
        loaded_trainer = c_util.load_trainer(saved_trainer_path)
        energy_fn_template = loaded_trainer.reference_energy_fn_template #test difference without template
        energy_params = loaded_trainer.params

    trajectory_generator = trajectory_generator_init(simulator_template,
                                                    energy_fn_template,
                                                    timings)


    #compute trajectory and quantities
    print(f'Start simulation with timestep {dt}')
    t_start = time.time()
    traj_state = trajectory_generator(energy_params, reference_state)
    print('ps/min: ', total_time / ((time.time() - t_start) / 60.))

    assert not traj_state.overflow, ('Neighborlist overflow during trajectory '
                                    'generation. Increase capacity and re-run.')

    compute_fns['energy'] = custom_quantity.energy_wrapper(energy_fn_template)
    compute_fns['kinetic_energy'] = custom_quantity.kinetic_energy_wrapper
    # compute_fns['kinetic_energy'] = custom_quantity.kinetic_energy_tensor
    compute_fns['total_energy'] = custom_quantity.total_energy_wrapper(energy_fn_template)
    #add kinetic energy


    print('Start quantity calculation')
    t_start2 = time.time()
    quantity_trajectory = quantity_traj(traj_state, compute_fns, energy_params)
    print('time for quantities: ', (time.time() - t_start2) / 60.)

    #plotting and prints
    if 'energy' in quantity_trajectory:
        energy_traj = quantity_trajectory['energy']
        # onp.savetxt(f'output/energy/{plotname}/pot_energy_{dt}.csv', energy_traj)
        print('energy max:',energy_traj[0],jnp.max(energy_traj),jnp.min(energy_traj))
        kinetic_energy_traj = quantity_trajectory['kinetic_energy']
        total_energy_traj = quantity_trajectory['total_energy']
        energies.append(energy_traj)
        kinetic_energies.append(kinetic_energy_traj)
        total_energies.append(total_energy_traj)
        # we assume samples are iid here: Is approximately true as we only save
        # configurations every ca 100 time steps
        x = jnp.linspace(0.0, total_time-t_equilib, int((total_time-t_equilib)/print_every))
        plt.figure()
        plt.plot(x, energy_traj, label='potential')
        plt.plot(x, kinetic_energy_traj, label=' kinetic')
        plt.plot(x, total_energy_traj, label='total')
        # plt.plot(trainer.gradient_norm_history)
        # plt.yscale('log')
        plt.legend()
        plt.title(f'Timestep {dt} ps')
        plt.ylabel('Energy [kJ $\mathrm{mol^{-1}}$]')
        plt.xlabel('Time [ps]')
        plt.savefig(f'output/energy/{plotname}/energies_'+f'{dt}dt.png')
        plt.show()       


plt.figure()
for e in range(len(energies)):
    plt.plot(x, energies[e], label=f'{time_array[e]} ps')
plt.legend()
plt.title('Potential Energies')
plt.ylabel('Energy  [kJ $\mathrm{mol^{-1}}$]')
plt.xlabel('Time [ps]')
plt.savefig(f'output/energy/{plotname}/pot_energy.png')
plt.show()

plt.figure()
for k in range(len(energies)):
    plt.plot(x, kinetic_energies[k], label=f'{time_array[k]} ps')
plt.legend()
plt.title('Kinetic Energies')
plt.ylabel('Energy  [kJ $\mathrm{mol^{-1}}$]')
plt.xlabel('Time [ps]')
plt.savefig(f'output/energy/{plotname}/kinetic_energy.png')
plt.show()

plt.figure()
for t in range(len(total_energies)):
    plt.plot(x, total_energies[t], label=f'{time_array[t]} ps')
plt.legend()
plt.title('Total Energies')
plt.ylabel('Energy  [kJ $\mathrm{mol^{-1}}$]')
plt.xlabel('Time [ps]')
plt.savefig(f'output/energy/{plotname}/total_energy.png')
plt.show() 