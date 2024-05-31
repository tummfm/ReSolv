"""Calculates RDF, ADF, TCF of provided trajectory. And also loads them for
plotting.
"""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import jax.numpy as jnp
# config.update("jax_debug_nans", True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

from jax_md import dataclasses, space, partition
from jax_md.util import Array
from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform, visible_device)

from chemtrain.traj_util import quantity_traj, TrajectoryState
from chemtrain.jax_md_mod import custom_space
from util import Postprocessing, Initialization
from util.visualization import plot_initial_and_predicted
from aa_simulations.paper_plots import plot_rdf_adf_tcf
import cloudpickle as pickle
from chemtrain import util
from jax.tree_util import tree_map
import time

plotname = 'testing_10fs_rand0'
# file = '../aa_simulations/confs/conf_O_10k.npy'
# file = '../aa_simulations/confs/conf_SPC_1k.npy'
# file = '../aa_simulations/confs/conf_COM_10k_final.npy'
# file2 = f'output/water_confs/confs_water_RE_{plotname}.npy'
file2 = f'output/water_confs/confs_water_RE_8k_4fs_280ps.npy'
# file = f'output/water_confs/confs_water_RE_8k_2fs_rand1111_batchref_1ns.npy'
file = f'output/water_confs/confs_water_FM_8k_2fs_100e_rand4444_1ns.npy'
# file = '../aa_simulations/confs/conf_SPC.npy'
# boxfile = '../aa_simulations/confs/length_COM_10k_shifted.npy'
boxfile = '../aa_simulations/confs/length_COM_10k_final.npy'


use_saved_properties = True
saved_name1 = 'FM_8k_10fs_final_rand0_1ns'
saved_name2 = 'RE_8k_10fs_final_rand0_1ns'
labels = ['Reference','FM','RE']
# labels = ['Reference','RE 4fs rand 1111']

# target_rdf = 'LJ'
# target_rdf = 'SPC'
# target_rdf = 'SPC_FW'
# target_rdf = 'Water_Ox'
target_rdf = 'TIP4P/2005'
rdf_struct = Initialization.select_target_rdf(target_rdf)
adf_struct = Initialization.select_target_adf('TIP4P/2005', 0.318)
tcf_struct = Initialization.select_target_tcf('TIP4P/2005', 0.5, nbins=50)

# add all target values here, target is only dummy
# target_dict = {'rdf': rdf_struct, 'adf': adf_struct}
target_dict = {'rdf': rdf_struct, 'adf': adf_struct, 'tcf': tcf_struct}
# target_dict = {'adf': adf_struct}
# target_dict = {'rdf': rdf_struct}
# target_dict = {'tcf': tcf_struct}
# target_dict = {'pressure': 1.}

###############################

# position_data = jnp.array(onp.load(file))
# used_dataset_size = 50
# subsampling = 1
# position_data_ref = get_dataset(file, retain=used_dataset_size,
#                                  subsampling=subsampling)
position_data = jnp.array(onp.load(file))
# position_data = onp.load(file)
# position_data2 = jnp.array(onp.load(file2))
pos_shape = position_data.shape
print('Data size:',pos_shape)
# print(position_data2.shape)

box_length = jnp.load(boxfile) #load box length
# box_length = 3.
box = jnp.array([box_length, box_length, box_length])
print('box:',box)

# position_data = scale_dataset_fractional(position_data, box)
R_init = position_data[0]
# R_init2 = position_data2[0]
print('first state',R_init.shape)

box_tensor, _ = custom_space.init_fractional_coordinates(box)

displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)
r_cut = 0.5
neighbor_fn = partition.neighbor_list(displacement, box[0], r_cut, dr_threshold=0.05,
                                        capacity_multiplier=2., fractional_coordinates=True)
nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)

compute_fns, _ = Initialization.build_quantity_dict(R_init, box_tensor, displacement, None,
                                                        nbrs_init, target_dict, init_class=None)

# nbrs_init2 = neighbor_fn.allocate(R_init2, extra_capacity=0)

# compute_fns2, _ = Initialization.build_quantity_dict(R_init2, box_tensor, displacement, None,
#                                                         nbrs_init2, target_dict, init_class=None)

#--------------------------- INITIALIZE ---------------------------------------#
# define pseudo state (with positions) for RDF and ADF functions and tuple (last sim state, nbrs)

@dataclasses.dataclass
class PseudoState:
    position: Array
 
traj = PseudoState(position_data)
last_state = (PseudoState(R_init),nbrs_init)
traj_state = TrajectoryState(sim_state=last_state, trajectory=traj)

# traj2 = PseudoState(position_data2)
# last_state2 = (PseudoState(R_init2),nbrs_init2)
# traj_state2 = TrajectoryState(sim_state=last_state2, trajectory=traj2)
if use_saved_properties is not None:
    quantity_trajectory = {'rdf':1,'adf':1,'tcf':1}
else:
    t_start = time.time()
    quantity_trajectory = quantity_traj(traj_state, compute_fns, batch_size=None)
    print('seconds: ', time.time() - t_start)
    # quantity_trajectory2 = quantity_traj(traj_state2, compute_fns2, vmap_batch_number=10)
    # print(quantity_trajectory['rdf'].shape)
quantity_trajectory = {'rdf':1,'adf':1,'tcf':1}

# plotting and prints
from pathlib import Path
Path('output/figures').mkdir(parents=True, exist_ok=True)
if 'rdf' in quantity_trajectory:
    if use_saved_properties is not None:
        computed_RDF = jnp.load(f'output/properties/{saved_name1}_RDF.npy')[:,1]
        computed_RDF2 = jnp.load(f'output/properties/{saved_name2}_RDF.npy')[:,1]
    else:
        computed_RDF = jnp.mean(quantity_trajectory['rdf'], axis=0)
        computed_RDF2 = None
        # computed_RDF2 = jnp.mean(quantity_trajectory2['rdf'], axis=0)
        jnp.save(f"output/properties/{plotname}_RDF",
                                 jnp.array([rdf_struct.rdf_bin_centers, computed_RDF]).T)
        # jnp.save(f"output/properties/RE_{plotname}_RDF",
                                # jnp.array([rdf_struct.rdf_bin_centers, computed_RDF2]).T)
    
    print(computed_RDF.shape)
    # np.savetxt(f"../aa_simulations/data/RDF/{plotname}_RDF.csv",
    #                             np.array([rdf_struct.rdf_bin_centers, computed_RDF]).T)
    plot_initial_and_predicted(rdf_struct.rdf_bin_centers, computed_RDF,
                                'RDF_compare_'+plotname, g_average_init=computed_RDF2,
                                reference=rdf_struct.reference, labels=labels,
                                axis_label=['RDF','r in $\mathrm{nm}$'])

if 'adf' in quantity_trajectory:
    if use_saved_properties is not None:
        computed_ADF = jnp.load(f'output/properties/{saved_name1}_ADF.npy')[:,1]
        computed_ADF2 = jnp.load(f'output/properties/{saved_name2}_ADF.npy')[:,1]
    else:
        computed_ADF = jnp.mean(quantity_trajectory['adf'], axis=0)
        computed_ADF2 = None
        # computed_ADF2 = jnp.mean(quantity_trajectory2['adf'], axis=0)
        jnp.save(f"output/properties/{plotname}_ADF",
                                    jnp.array([adf_struct.adf_bin_centers, computed_ADF]).T)
        # jnp.save(f"output/properties/RE_{plotname}_ADF",
                                    # jnp.array([adf_struct.adf_bin_centers, computed_ADF2]).T)
    plot_initial_and_predicted(adf_struct.adf_bin_centers, computed_ADF,
                                'ADF_compare_'+plotname, g_average_init=computed_ADF2,
                                reference=adf_struct.reference, labels=labels,
                                axis_label=['ADF',r'$\alpha$ in $\mathrm{rad}$'])

if 'tcf' in quantity_trajectory:
    if use_saved_properties is not None:
        equilateral = jnp.load(f'output/properties/{saved_name1}_TCF.npy')[:,1]
        equilateral2 = jnp.load(f'output/properties/{saved_name2}_TCF.npy')[:,1]
    else:
        computed_TCF = jnp.mean(quantity_trajectory['tcf'], axis=0)
        # computed_TCF2 = jnp.mean(quantity_trajectory2['tcf'], axis=0)
        #slice and print equilaterial
        equilateral = jnp.diagonal(jnp.diagonal(computed_TCF))
        equilateral2 = None
        # equilateral2 = jnp.diagonal(jnp.diagonal(computed_TCF2))
        jnp.save(f"output/properties/{plotname}_TCF",
                                    jnp.array([tcf_struct.tcf_x_bin_centers[0,:,0], equilateral]).T)
        # jnp.save(f"output/properties/RE_{plotname}_TCF",
        #                             jnp.array([tcf_struct.tcf_x_bin_centers[0,:,0], equilateral2]).T)
    plot_initial_and_predicted(tcf_struct.tcf_x_bin_centers[0,:,0], equilateral,
                                    'TCF_compare_'+plotname, g_average_init=equilateral2,
                                    reference=tcf_struct.reference, labels=labels,
                                    axis_label=['TCF','r in $\mathrm{nm}$'])


plot_rdf_adf_tcf([rdf_struct.rdf_bin_centers, adf_struct.adf_bin_centers, tcf_struct.tcf_x_bin_centers[0,:,0]],
                    [computed_RDF, computed_ADF, equilateral],'1_compare_'+plotname,
                    g_average_init_list=[computed_RDF2, computed_ADF2, equilateral2], reference_list=[rdf_struct.reference,
                    adf_struct.reference, tcf_struct.reference], labels=labels)