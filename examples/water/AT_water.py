"""Currently only forward simulation of AT water model."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)

import time

from chemtrain.traj_util import process_printouts, trajectory_generator_init, \
    quantity_traj
from chemtrain.jax_md_mod import io
from util import Postprocessing, Initialization

file = 'data/confs/Water_AT_2_5nm.gro'
# file = 'data/confs/Water_AT_2nm.gro'

model = 'CGDimeNet'

# saved_trainer_path = '../notebooks/saved_models/CG_water_GNN.pkl'
saved_trainer_path = 'output/difftre/trained_model.pkl'
saved_trainer_path = None

kbT = 2.49435321
time_step = 0.001  # For SPC/FW 1fs time step necessary

total_time = 30.
t_equilib = 5.
print_every = 0.1

target_rdf = 'Water_Ox'
rdf_struct = Initialization.select_target_rdf(target_rdf)
adf_struct = Initialization.select_target_adf('Water_Ox', 0.318)

# add all target values here, target is only dummy
target_dict = {'rdf': rdf_struct, 'adf': adf_struct, 'pressure': 1.}
# TODO adjust targets to NPT and multiple species

###############################

box, R, masses, species = io.load_box(file)
simulation_data = Initialization.InitializationClass(
    R_init=R, box=box, kbT=kbT, masses=masses, dt=time_step, species=species)
timings = process_printouts(time_step, total_time, t_equilib, print_every)

reference_state, energy_params, simulation_fns, compute_fns, targets = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         target_dict,
                                         wrapped=True,  # bug otherwise
                                         integrator='Nose_Hoover')

simulator_template, energy_fn_template, neighbor_fn = simulation_fns

trajectory_generator = trajectory_generator_init(simulator_template,
                                                 energy_fn_template,
                                                 timings)

# compute trajectory and quantities
t_start = time.time()
traj_state = trajectory_generator(energy_params, reference_state)
print('ps/min: ', total_time / ((time.time() - t_start) / 60.))