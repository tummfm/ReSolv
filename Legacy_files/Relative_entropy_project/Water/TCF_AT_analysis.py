import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import jax.numpy as jnp
# from jax.config import config
# config.update('jax_debug_nans', True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)

from jax import jit, numpy as jnp
from jax_md import space, util, dataclasses, partition

import numpy as onp

import chemtrain.util as c_util
from chemtrain.traj_util import TrajectoryState, quantity_traj
from chemtrain.jax_md_mod import custom_quantity
from chemtrain.jax_md_mod.custom_space import _rectangular_boxtensor

from util.visualization import plot_initial_and_predicted

import time

Array = util.Array

#Parameters
plotname='test_TCF_eq_1k_50b_0.6'
typename = 'TCF'
configuration_str = '../aa_simulations/confs/conf_COM_10k_final.npy'
transparent = True #transparent plotbackground
split = False #Split trajectory
split_into = 10
used_dataset_size = 1000
subsampling = 1 #take every n snapsshot
r_cut = 0.6
TCF_start = 0.2
nbins = 50

dx_bin = (r_cut - TCF_start) / float(nbins)
#TODO: bin centers needed for plotting, provide in tcf_struct?, otherwise not needed?
tcf_bin_centers = jnp.linspace(TCF_start + dx_bin / 2.,
                                   r_cut - dx_bin / 2.,
                                   nbins)

sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers, tcf_z_bin_centers = \
                    custom_quantity.tcf_discretization(r_cut, nbins,TCF_start)

tcf_struct = custom_quantity.TCFParams(None, sigma_tcf, volume, tcf_x_binx_centers,
                    tcf_y_bin_centers, tcf_z_bin_centers)

box_length = 3.0
box_length = jnp.array(jnp.load('../aa_simulations/confs/length_COM_10k_final.npy')) #unnecessary?
# box_length = jnp.array(jnp.load('../aa_simulations/confs/length_COM_1k_final.npy'))
box = jnp.array([box_length, box_length, box_length])
box_tensor = _rectangular_boxtensor(box)

displacement, shift = space.periodic(box)

# position_data = jnp.array(jnp.load('../aa_simulations/confs/conf_atoms_LJ_27.npy'))
# position_data = jnp.array(jnp.load('../aa_simulations/confs/conf_COM_10k_final.npy'))
# position_data = jnp.array(jnp.load('../aa_simulations/confs/conf_COM_1k_final.npy'))
# position_data = jnp.array(jnp.load('../aa_simulations/confs/conf_SPC_1k.npy'))
# position_data = shift(position_data,0) #shift all COMs which are out of box, might not be necessary

position_data = c_util.get_dataset(configuration_str, retain=used_dataset_size,
                                 subsampling=subsampling)
print('Dataet size:',position_data.shape)

R_init = position_data[0]
# position_data = onp.array(position_data)

blocks = jnp.split(position_data, split_into)
n_blocks = len(blocks)
print("Number of blocks:", n_blocks)
print("Size of block:", blocks[0].shape)

n_particles = R_init.shape[0]
n_snaps = position_data.shape[0]
print('N particles',n_particles)

neighbor_fn = partition.neighbor_list(displacement, box[0], r_cut, dr_threshold=0.05,
                                        capacity_multiplier=2., fractional_coordinates=False)
nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)

max = nbrs_init.max_occupancy
print('max nbrs:',max)
print('max theoretical triplets:',(max)**2*R_init.shape[0])

tcf_fn = custom_quantity.init_tcf_nbrs(displacement, tcf_struct, box_tensor,
                                            nbrs_init=nbrs_init, batch_size=1000) #test this
# tcf_fn = jit(tcf_fn)

compute_fns = {}
compute_fns['tcf'] = tcf_fn

@dataclasses.dataclass
class PseudoState:
    position: Array

if split:
    print('Using blocks:')
    for i in range(n_blocks):
        traj = PseudoState(blocks[i])
        R = PseudoState(R_init)
        last_state = (R,nbrs_init)
        traj_state = TrajectoryState(sim_state=last_state, trajectory=traj)

        t_start = time.time()
        quantity_trajectory = quantity_traj(traj_state, compute_fns)
        print(f'time iteration {i}: ', time.time() - t_start,'s')

        computed_TCF = jnp.mean(quantity_trajectory['tcf'], axis=0)

        #slice and print equilaterial
        equilateral = jnp.diagonal(jnp.diagonal(computed_TCF))

        plot_initial_and_predicted(tcf_bin_centers, equilateral, typename,
                                                        plotname+f'_block_{i}', transparent=transparent)

        jnp.save('output/TCF/'+plotname+f'_block_{i}',computed_TCF)

else:
        print('Using whole dataset')
        traj = PseudoState(position_data)
        R = PseudoState(R_init)
        last_state = (R,nbrs_init)
        traj_state = TrajectoryState(sim_state=last_state, trajectory=traj)

        t_start = time.time()
        quantity_trajectory = quantity_traj(traj_state, compute_fns)
        print(f'time iteration: ', time.time() - t_start,'s')

        computed_TCF = jnp.mean(quantity_trajectory['tcf'], axis=0)

        #slice and print equilaterial
        equilateral = jnp.diagonal(jnp.diagonal(computed_TCF))

        plot_initial_and_predicted(tcf_bin_centers, equilateral, typename,
                                                        plotname, transparent=transparent)

        jnp.save('output/TCF/'+plotname,computed_TCF)