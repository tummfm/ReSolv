import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import pickle
import jax.numpy as jnp
from jax_md import partition, simulate, space, quantity
from jax import random, lax, jit, vmap
import numpy as onp
import haiku as hk

from chemtrain.jax_md_mod import custom_space, custom_quantity
from chemtrain import neural_networks, sparse_graph
from functools import partial


# Temperature
system_temperature = 300.0  # 298.15K = Kelvin = 25 deg. celsius
Boltzmann_constant = 0.0083145107  # in kJ / mol K
kbT = system_temperature * Boltzmann_constant

convert_from_GPa_to_kJ_mol_nm_3 = 10**3 / 1.66054

# # Timing
total_time_long = 10.0   # 11000=11ns
dt = 0.5e-3
steps = int(total_time_long // dt)
write_every = int(.1 / dt)  # -> =0.1ps

# Loaded model
use_best_params = True

num_epochs = 70
load_trainer_path = "ANI1xDB/100222_ANI1x_subset_First1000_"+str(num_epochs)+"epochs_Trial.pkl"


# # Load trainer and energy function
# with open(load_trainer_path, 'rb') as pickle_file:
#     trainer_loaded = pickle.load(pickle_file)
#     energy_fn = trainer_loaded.energy_fn

# Load sample
# Create 100nm^3 box -> Use large box to avoid periodic effects
load_dataset_name = '../../../../FreeEnergySolubility/examples/FreeEnergyScripts/ANI1xDB/'

pad_pos = onp.load(load_dataset_name + 'pad_pos.npy')[:100]
pad_pos *= 0.1
pad_species = onp.load(load_dataset_name + 'mask_species.npy')[:100]
box = jnp.eye(3)*100
scale_fn = custom_space.fractional_coordinates_triclinic_box(box)

# Take padded positions. -> Set padded positions such that energy and force contribution is zero.
#  1. Add to non zero values 50nm
#  2. Set padded pos at x=0.6, 1.2, 1.8, .. [nm], y = z = 0. -> Energy contribution is zero
for i, pos in enumerate(pad_pos):
    pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])

for i, pos in enumerate(pad_pos):
    x_spacing = 0
    for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
        x_spacing += 0.6
        pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing

# Scale to fractional
pad_pos = lax.map(scale_fn, pad_pos)
pad_species = onp.array(pad_species, dtype='int32')
sample_pos = jnp.array(pad_pos[0])


# Set up single molecule to run nvt simulation

mol_len = onp.count_nonzero(pad_species[0])
box_tensor = box
R_init = pad_pos[0][:mol_len]
species_init = pad_species[0][:mol_len]
mass = []
for i in species_init:
    if i == 1:
        mass.append(1.008)
    elif i == 6:
        mass.append(12.011)
    else:
        raise NotImplementedError
mass = jnp.array(mass)

print("Debug")


N = R_init.shape[0]
displacement, shift = space.periodic_general(box_tensor)

# define random seed for initialization of model and simulation
key = random.PRNGKey(0)
model_init_key, simulation_init_key = random.split(key, 2)

# Neighbors
r_cut_nbrs = 0.5
neighbor_fn = partition.neighbor_list(displacement, box_tensor, r_cut_nbrs,
                                      dr_threshold=0.05,
                                      capacity_multiplier=2.0,
                                      disable_cell_list=True,
                                      fractional_coordinates=True)
nbrs_init = neighbor_fn(R_init)

testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
            displacement, R_init, nbrs_init, 0.5)

# TODO, be aware that displacement has to change for every box! -> Have to pass box as kwarg
init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(displacement=displacement,
                                                             r_cutoff=r_cut_nbrs,
                                                             n_species=70,
                                                             positions_test=R_init,
                                                             neighbor_test=nbrs_init,
                                                             max_triplet_multiplier=2.5,
                                                             max_edge_multiplier=2.5,
                                                             embed_size=32)

# Load trainer
# Load trainer and energy function
with open(load_trainer_path, 'rb') as pickle_file:
    trainer_loaded = pickle.load(pickle_file)

if use_best_params:
    init_params = trainer_loaded.best_params
    fn = lambda module_name, name, arr: jnp.array(arr)
    init_params = hk.data_structures.map(fn, init_params)
else:
    init_params = trainer_loaded.params


# Define Energy function - DimeNet++
def energy_fn_template(energy_params):
    gnn_energy = partial(GNN_energy, energy_params)
    def energy(R, neighbor, **dynamic_kwargs):
        return gnn_energy(R, neighbor=neighbor, **dynamic_kwargs)
    return energy
energy_fn_init = energy_fn_template(init_params)

# init, apply = simulate.npt_nose_hoover(energy_fn=energy_fn_init, shift_fn=shift, dt=dt, pressure=0., kT=kbT)
init, apply = simulate.nvt_langevin(energy_or_force=energy_fn_init, shift=shift, dt=dt, kT=kbT, gamma=4.)


state = init(key=simulation_init_key, R=R_init, mass=mass, neighbor=nbrs_init)

# TODO -> Define step_fn as done in Jax-MD NPT example.
compute_stress = custom_quantity.init_virial_stress_tensor(energy_fn_template=energy_fn_template)
compute_pressure = custom_quantity.init_pressure(energy_fn_template=energy_fn_template)
compute_pressure_without_kinetic = custom_quantity.init_pressure(energy_fn_template=energy_fn_template, include_kinetic=False)

@jit
def step_fn(i, state_nbrs_log):
    state, nbrs, log = state_nbrs_log

    # Log information on simulation # ATTENTION: Be aware that in newer jax-md versions velocity -> momentum is used.
    m_v = vmap(lambda x, y: x*y)(mass, state.velocity)
    dump_mass = 1.
    T = quantity.temperature(velocity=m_v, mass=dump_mass)
    log['kT'] = lax.cond(i % write_every == 0,
                         lambda temp: temp.at[i // write_every].set(T),
                         lambda temp: temp,
                         log['kT'])


    # Write out positions
    log['pos'] = lax.cond(i % write_every == 0,
                          lambda p: p.at[i // write_every].set(state.position),
                          lambda p: p,
                          log['pos'])

    # Simulation step
    state = apply(state, neighbor=nbrs)
    nbrs = nbrs.update(state.position)
    return state, nbrs, log


log = {
    'kT': onp.zeros(((steps // write_every)+1,),),
    'P': onp.zeros(((steps // write_every)+1,),),
    'P_no_kin': onp.zeros(((steps // write_every)+1,),),
    'pos': onp.zeros(((steps // write_every)+1,)+R_init.shape)
}
state, nbrs, log = lax.fori_loop(0, steps+1, step_fn, (state, nbrs_init, log))

print(nbrs.did_buffer_overflow)
print('kT: ', log['kT'] / Boltzmann_constant)
print('Pos: ', log['pos'])