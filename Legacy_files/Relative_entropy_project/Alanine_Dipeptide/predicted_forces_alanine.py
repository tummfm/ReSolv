"""Evaluates forces on testset for Alanine."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import jax.numpy as jnp
# config.update('jax_debug_nans', True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform, visible_device)

import numpy as onp

from jax import lax, value_and_grad

from chemtrain.jax_md_mod import io
from jax_md import util, space, partition
from chemtrain.jax_md_mod import custom_space
from chemtrain.util import scale_dataset_fractional
from chemtrain.traj_util import process_printouts
from chemtrain import util
import cloudpickle as pickle
from jax.tree_util import tree_map
from util import Initialization
      
# model = 'LJ'
# model = 'Tabulated'
model = 'CGDimeNet'

save_name = 'RE_150up_2fs_norewe_60k'
save_path = 'plots/postprocessing/force_scatter_'
# plotname = 'FM_1k_vt_1e's

# saved_trainer_path = 'notebooks/saved_models/CG_water_GNN.pkl'
# saved_trainer_path = '../examples/output/difftre/trained_model.pkl'
# saved_trainer_path = '../examples/output/force_matching/trained_model_8k_1e.pkl'
# saved_trainer_path = '../examples/output/rel_entropy/trained_model_1k_70ps_test.pkl'
# saved_trainer_path = '../examples/output/force_matching/trained_model_8k_20e_testing_new_10.pkl'
saved_trainer_path = None

saved_params_path = '../examples/output/rel_entropy/trained_params_alanine_heavy_RE_2fs_1ns_150up_lr0.03_lrd150_d01_norewe.pkl'

mapping = 'heavy'
configuration_str = f'alanine_dipeptide/confs/confs_{mapping}_100ns.npy'
force_str = f'alanine_dipeptide/confs/forces_{mapping}_100ns.npy'
file_topology = f'../examples/data/confs/Alanine_dipeptide_{mapping}_2_7nm.gro'
used_dataset_size = 440000
subsampling  = 1

box, R, masses, species, bonds = io.load_box(file_topology) # initial configuration


position_data = util.get_dataset(configuration_str, retain=used_dataset_size,
                                 subsampling=subsampling)

position_data2, test_data = util.get_dataset(configuration_str, retain=used_dataset_size,
                                 subsampling=subsampling,test_split=True)

print('test size:',test_data.shape)        
print(onp.array_equal(position_data,position_data2))
# print(onp.array_equal(position_data,position_data3))

force_data = util.get_dataset(force_str, retain=used_dataset_size,
                              subsampling=subsampling)

force_data2, test_forces = util.get_dataset(force_str, retain=used_dataset_size,
                              subsampling=subsampling, test_split=True)

test_data = test_data[:-1] #100k test set instead of 1000001
test_forces = test_forces[:-1]

# jnp.save('alanine_dipeptide/forces/test_forces_alanine_60k',test_forces)
print('test size:',test_data.shape)
print('test forces:',test_forces.shape)
#for initialization
R = position_data[0]

position_data = scale_dataset_fractional(position_data, box)
test_data = scale_dataset_fractional(test_data, box)
R_init = test_data[0]
print('first state',R_init.shape)

# pressure = onp.load('../examples/output/surface_tension/pressure_tensor_ST_test3.npy')
# print('pressure',pressure.shape)
# print('pressure',pressure[0])

# def calc_surface_tension(pressure, box_length):
#     '''Calculates the surface tension on the pressure tensor trajectory'''
#     #TODO: Move into right file
#     pT = 0.5*(pressure[:,0,0]-pressure[:,1,1])
#     pN = pressure[:,2,2]
#     surface_tension = jnp.mean(pN-pT, axis=0)/3*box_length
#     return surface_tension

# print('surface tension', calc_surface_tension(pressure,box_length))

box_tensor, _ = custom_space.init_fractional_coordinates(box)

displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)
r_cut = 0.5

neighbor_fn = partition.neighbor_list(displacement, box, r_cut, dr_threshold=0.05, capacity_multiplier=1.,
                                                        fractional_coordinates=True, disable_cell_list=True)
nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)

if saved_trainer_path is not None:
    print('using loaded trainer')
    loaded_trainer = util.load_trainer(saved_trainer_path)
    energy_fn_template = loaded_trainer.reference_energy_fn_template #test difference without template
    energy_params = loaded_trainer.params
    energy_fn = energy_fn_template(energy_params)

if saved_params_path is not None:
    print('using saved params')
    #repulsive prior (sigma, epsilon, r_cut)
    prior_dict = {'bond': 1., 'angle': 1., 'dihedral': 1.}
    idxs, constants = Initialization.select_protein('heavy_alanine_dipeptide', prior_dict)

    kbT = 2.49435321
    time_step = 0.002  # Bigger time_step possible for CG water?
    total_time =  75.
    t_equilib = 5.
    print_every = 0.1
    target_dict = {}

    simulation_data = Initialization.InitializationClass(R_init=R,
        box=box, kbT=kbT, masses=masses, dt=time_step, species=species)
    timings = process_printouts(time_step, total_time, t_equilib, print_every)

    _, _, simulation_fns, _, _ = \
        Initialization.initialize_simulation(simulation_data,
                                            model,
                                            target_dict,
                                            wrapped=True,  # bug otherwise
                                            integrator='Langevin',
                                            prior_idxs=idxs,
                                            prior_constants=constants)

    _, energy_fn_template, _ = simulation_fns

    with open(saved_params_path, 'rb') as pickle_file:
            params = pickle.load(pickle_file)
            energy_params = tree_map(jnp.array, params)
            
    energy_fn = energy_fn_template(energy_params)

print('reference forces size:', force_data.shape)

def init_single_prediction(nbrs_init, energy_fn):
    '''Initialize predictions for a single snapshot. Can be used to
    parametrize potentials from per-snapshot energy, force and/or virial.
    '''
    def single_prediction(positions):
        # TODO check for neighborlist overflow and hand through
        nbrs = nbrs_init.update(positions)
        energy, negative_forces = value_and_grad(energy_fn)(positions, neighbor=nbrs)
        predictions = {'U': energy, 'F': -negative_forces}
        return predictions
    return single_prediction

single_prediction = init_single_prediction(nbrs_init, energy_fn)

predictions = lax.map(single_prediction, test_data)
##use of lax.map instead of vmap to avoid out of memory -> or use batches?
print(predictions['F'].shape)

print('Precited energies vs valueandgrad function:', jnp.mean(predictions['U'])) # same
print('Precited forces vs valueandgrad function:', jnp.max(predictions['F']), jnp.min(predictions['F']))
print('reference forces', jnp.max(force_data),jnp.min(force_data))

jnp.save('alanine_dipeptide/forces/predicted_forces_alanine_'+save_name,predictions['F'])
# jnp.save('alanine_dipeptide/forces/test_forces_2k',test_forces)
