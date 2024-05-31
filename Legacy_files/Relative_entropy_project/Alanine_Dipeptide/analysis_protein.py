import os
import sys
if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import jax.numpy as jnp
# config.update('jax_debug_nans', True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform,visible_device)

import numpy as onp

from jax import vmap

from chemtrain.jax_md_mod import io
from jax_md import space
from chemtrain.jax_md_mod import custom_space
from chemtrain.util import scale_dataset_fractional, load_trainer
from util import visualization
import aa_simulations.aa_util as aa


save_name = '40x100ns'
folder_name = f'models_40x10ns/1D_free_energy/{save_name}/'
folder_name2 = f'1D_free_energy/'
split = 40
plots = False
################################################################################
#Load confs
# positions = onp.load('../examples/output/surface_tension/'+'
#                                                   confs_FM_8k_60ps_1ns.npy')
# positions = onp.load('../examples/output/alanine_confs/'+
                    # 'confs_alanine_RE_stepwise_15x10ns_2.npy')
# positions = onp.load('../examples/output/alanine_confs/'+
                    # 'confs_test_alanine_FM_both_prior_100gamma_002_10ns.npy')
# positions = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_stepwise_15x10ns_2.npy')
# positions = onp.load('../examples/output/alanine_confs/'+
                                        # 'confs_test_alanine_FM_10e_lj_10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_stepwise_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_100upx100ps_nolj_40x100ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_3C_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_100upx100ps_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_nodihedral_3C_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_dihedral_3C_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_nolj_3C_40x10ns.npy')

# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_nodihedral_3C_40x10ns.npy')

# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_nolj_3C_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
                    # 'confs_alanine_FM_nolj_3C_40x10ns_3.npy')

# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_dihedral_3C_40x10ns.npy')

# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_dihedral_40x10ns.npy')

# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_nod_3C_40x100ns.npy')
# positions = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_nolj_3C_40x100ns.npy')
                
# positions2 = onp.load('../examples/output/alanine_confs/'+
                    # 'confs_alanine_FM_10e_batch200_40x100ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_300upx100ps_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_RE_40upx100ps_60upx500ps_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_10e_5klj_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_5e_batch512_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_10e_batch512_40x10ns.npy')
# positions2 = onp.load('../examples/output/alanine_confs/'+
#                     'confs_alanine_FM_10e_batch200_40x10ns.npy')
# print(positions2.shape)
# print(positions2)
positions_ref = onp.load('alanine_dipeptide/confs/confs_heavy_100ns.npy')
# positions = onp.load('../examples/output/surface_tension/'+'
#                                                   confs_RE_8k_300up_1ns.npy')
# positions = onp.load('../examples/output/surface_tension/'+'
#                                                       confs_SPC_1k_1ns.npy')
# positions = onp.load('../examples/output/surface_tension/'+
#                                                   'confs_FM_virial_1ns.npy')
# print('Positions:',positions.shape)
# if positions.ndim > 3:
#     positions = positions.reshape((-1,10,3))
#     print('Positions reshaped:',positions.shape)
################################################################################
#Create Positions, Forces
# positions3 = io.trr_to_numpy('alanine_dipeptide/data/100ns/heavy_first.gro',
#                                 'alanine_dipeptide/data/100ns_random_seed/100ns_3/heavyMD_100ns_r3.trr',
#                                                                 force=False)
# positions4 = io.trr_to_numpy('alanine_dipeptide/data/100ns/heavy_first.gro',
#                                 'alanine_dipeptide/data/100ns_random_seed/100ns_4/heavyMD_100ns_r4.trr',
#                                                                 force=False)
# positions5 = io.trr_to_numpy('alanine_dipeptide/data/100ns/heavy_first.gro',
#                                 'alanine_dipeptide/data/100ns_random_seed/100ns_5/heavyMD_100ns_r5.trr',
#                                                                 force=False)
# positions6 = io.trr_to_numpy('alanine_dipeptide/data/100ns/heavy_first.gro',
#                                 'alanine_dipeptide/data/100ns_random_seed/100ns_6/heavyMD_100ns_r6.trr',
#                                                                 force=False)
# positions7 = io.trr_to_numpy('alanine_dipeptide/data/100ns/heavy_first.gro',
#                                 'alanine_dipeptide/data/100ns_random_seed/100ns_7/heavyMD_100ns_r7.trr',
#                                                                 force=False)
# positions2 = onp.load('alanine_dipeptide/confs/confs_heavy_100ns_r2.npy')
# positions3 = onp.load('alanine_dipeptide/confs/confs_heavy_100ns_r3.npy')
# positions4 = onp.load('alanine_dipeptide/confs/confs_heavy_100ns_r4.npy')
# positions5 = onp.load('alanine_dipeptide/confs/confs_heavy_100ns_r5.npy')
# positions6 = onp.load('alanine_dipeptide/confs/confs_heavy_100ns_r6.npy')
# positions7 = onp.load('alanine_dipeptide/confs/confs_heavy_100ns_r7.npy')
# print(positions.shape)
# onp.save('alanine_dipeptide/confs/confs_heavy_100ns_r3',positions3)
# onp.save('alanine_dipeptide/confs/confs_heavy_100ns_r4',positions4)
# onp.save('alanine_dipeptide/confs/confs_heavy_100ns_r5',positions5)
# onp.save('alanine_dipeptide/confs/confs_heavy_100ns_r6',positions6)
# onp.save('alanine_dipeptide/confs/confs_heavy_100ns_r7',positions7)
# onp.save('alanine_dipeptide/confs/forces_heavy_100ns_3',forces)
# print('Positions:',positions.shape)
# print('Forces:',forces.shape)
# print(positions[0],positions[-1])
# print(forces[0],forces[-1])
# ref = '_5'
positions2 = onp.load(f'alanine_dipeptide/confs/confs_heavy_100ns_2.npy')
positions3 = onp.load(f'alanine_dipeptide/confs/confs_heavy_100ns_3.npy')
positions4 = onp.load(f'alanine_dipeptide/confs/confs_heavy_100ns_4.npy')
positions5 = onp.load(f'alanine_dipeptide/confs/confs_heavy_100ns_5.npy')
# print(forces_ref[0],forces_ref[-1])
################################################################################
#Load forces
# model = 'Relative Entropy'
# forces = onp.load('results/forces/predicted_forces_RE_alanine_stepwise_100k.npy')
# # FM_forces = onp.load('results/forces/predicted_forces_FM_8k_1e.npy')
# test_forces = onp.load('results/forces/test_forces_alanine_100k.npy')
# print('Predicted forces shape:',forces.shape)
# print('Test forces shape:',test_forces.shape)
# if plots:
#     visualization.plot_scatter_forces(forces,test_forces,save_name,model,line=6000)
################################################################################
#Load harmonic constants
# eq_bond_length = onp.load('alanine_dipeptide/confs/'+
#                                   'Alanine_dipeptide_heavy_eq_bond_length.npy')
# eq_bond_variance = onp.load('alanine_dipeptide/confs/'+
#                                 'Alanine_dipeptide_heavy_eq_bond_variance.npy')

# eq_angle = onp.load('alanine_dipeptide/confs/'+
#                                         'Alanine_dipeptide_heavy_eq_angle.npy')
# eq_angle_variance = onp.load('alanine_dipeptide/confs/'+
#                                'Alanine_dipeptide_heavy_eq_angle_variance.npy')
################################################################################
#Load topology
system_temperature = 300 # Kelvin
Boltzmann_constant = 0.0083145107  # in kJ / mol K
kbT = system_temperature * Boltzmann_constant
# box_length = jnp.load('confs/length_COM_10k_final.npy') #load box length
# box = jnp.array([box_length, box_length, box_length])
#box = jnp.ones(3) #fractional coordiantes
box, R, masses, species, bonds = io.load_box(
                                f'alanine_dipeptide/data/100ns/heavy_first.gro')

box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)

# displacement_periodic, _ = space.periodic(box)
displacement, shift = space.periodic_general(box_tensor,
                                    fractional_coordinates=True, wrapped=True)

##########################################################
box2, _, _, _, _ = io.load_box(
            f'alanine_dipeptide/data/100ns_same_seed/100ns_2/heavy_first_2.gro')

box_tensor2, scale_fn2 = custom_space.init_fractional_coordinates(box2)

displacement2, shift2 = space.periodic_general(box_tensor2,
                                    fractional_coordinates=True, wrapped=True)
############################################################                             
box3, _, _, _, _ = io.load_box(
            f'alanine_dipeptide/data/100ns_same_seed/100ns_3/heavy_first_3.gro')

box_tensor3, scale_fn3 = custom_space.init_fractional_coordinates(box3)

displacement3, shift3 = space.periodic_general(box_tensor3,
                                    fractional_coordinates=True, wrapped=True)
############################################################ 
box4, _, _, _, _ = io.load_box(
            f'alanine_dipeptide/data/100ns_same_seed/100ns_4/heavy_first_4.gro')

box_tensor4, scale_fn4 = custom_space.init_fractional_coordinates(box4)

displacement4, shift4 = space.periodic_general(box_tensor4,
                                    fractional_coordinates=True, wrapped=True)
############################################################ 
box5, _, _, _, _ = io.load_box(
            f'alanine_dipeptide/data/100ns_same_seed/100ns_5/heavy_first_5.gro')

box_tensor5, scale_fn5 = custom_space.init_fractional_coordinates(box5)

displacement5, shift5 = space.periodic_general(box_tensor5,
                                    fractional_coordinates=True, wrapped=True)
############################################################ 

#Inverse to scale_fn
inv_scale_fn = lambda R: jnp.dot(R, box_tensor)
# print(positions[0,0])
# positions = inv_scale_fn(positions)
# print(positions[0,0])
# #If we want fractional coordinates
# positions2 = scale_dataset_fractional(positions2, box)
# positions3 = scale_dataset_fractional(positions3, box)
# positions4 = scale_dataset_fractional(positions4, box)
# positions5 = scale_dataset_fractional(positions5, box)
# positions6 = scale_dataset_fractional(positions6, box)
# positions7 = scale_dataset_fractional(positions7, box)
# positions2 = positions2[26:]
# positions3 = positions3[26:]
# positions4 = positions4[26:]
# positions5 = positions5[26:]
# positions6 = positions6[26:]
# positions7 = positions7[26:]
# print(positions2.shape)
# print(positions3.shape)
# print(positions4.shape)
# print(positions5.shape)
# print(positions6.shape)
# print(positions7.shape)
positions2 = scale_dataset_fractional(positions2, box2)
positions3 = scale_dataset_fractional(positions3, box3)
positions4 = scale_dataset_fractional(positions4, box4)
positions5 = scale_dataset_fractional(positions5, box5)
positions_ref = scale_dataset_fractional(positions_ref, box)
positions2 = positions2[1:]
positions3 = positions3[1:]
positions4 = positions4[1:]
positions5 = positions5[1:]
positions_ref = positions_ref[1:]
print(positions_ref.shape)
################################################################################
#Load trainer
# saved_trainer_path = ('../examples/output/rel_entropy/'
                            # 'trained_model_alanine_heavy_100up_15x100ps.pkl')
saved_trainer_path = None
# saved_init_path = ('../examples/output/rel_entropy/'
                            # 'init_points_alanine_heavy_100up_40x100_params.npy')
# saved_init_path = ('../examples/output/rel_entropy/'
                            # 'init_points_alanine_heavy_extra_100up_40x400_40k_params.npy')
# saved_init_path = None

# if saved_trainer_path is not None:
#     loaded_trainer = load_trainer(saved_trainer_path)
#     energy_fn_template = loaded_trainer.reference_energy_fn_template
#     energy_params = loaded_trainer.params

#     init_loaded = loaded_trainer.init_points[0]
#     init_positions = jnp.array(init_loaded)
#     print(init_positions.shape)

# elif saved_init_path is not None:
#     init_positions = jnp.load(saved_init_path)
#     print(init_positions.shape)

# else:
#     init_positions = None
################################################################################
#Load points from forward simulation
#TODO: combine into 1 file
#better naming and loading method
# saved_forward_path = ('../examples/output/init_points/'
#                             'RE_100upx40x100ps_25x10ns_chosen.npy')
# saved_forward_path2 = ('../examples/output/init_points/'
# #                             'RE_100upx40x100ps_25x10ns_chosen_reference.npy')
# saved_forward_path = ('../examples/output/init_points/'
#                             'RE_stepwise_10ns_reference.npy')
# saved_forward_path2 = ('../examples/output/init_points/'
#                             'RE_stepwise_10ns.npy.npy')
# saved_forward_path = None

# if saved_forward_path is not None:
#     init_ref_position = jnp.load(saved_forward_path2)
#     init_sim_position = jnp.load(saved_forward_path)
#     print('forward loaded', init_sim_position.shape)
#     # print(init_ref_position.shape)
    

################################################################################
# Protein Visualization (alanine dipeptide 10 heavy atoms)
# bond
# bond_lengths = aa.bond_length(bonds,positions,displacement)
# bond_lengths_ref = aa.bond_length(bonds,positions_ref,displacement)

# labels = ['CH3-C','C-O','C-N','CA-C','C-O','CA-CB','CA-N','C-N','N-CH3']
# print(jnp.mean(bond_lengths,axis=0),jnp.var(bond_lengths,axis=0))
# print(eq_bond_length,eq_bond_variance)
# if plots:
#     visualization.plot_bond_length(bond_lengths,save_name,bond_lengths_ref,
#                                                                     labels)

# #angle
# angle_idxs = jnp.array([[0,1,2],
#                        [0,1,3],
#                        [2,1,3],
#                        [1,3,4],
#                        [3,4,5],
#                        [3,4,6],
#                        [5,4,6],
#                        [4,6,7],
#                        [4,6,8],
#                        [7,6,8],
#                        [6,8,9],])

# mask = jnp.ones((angle_idxs.shape[0],1))
# angles = vmap(angle_triplets, (0,None,None,None))(positions,
#                                     displacement,angle_idxs,mask)*180/jnp.pi
# angles_ref = vmap(angle_triplets, (0,None,None,None))(positions_ref,
#                                     displacement,angle_idxs,mask)*180/jnp.pi

# print(jnp.mean(angles,axis=0),jnp.var(angles,axis=0))
# print(jnp.max(angles),jnp.min(angles))
# print(eq_angle,eq_angle_variance)
# print(angles.shape)
# if plots:
#     visualization.plot_angles(angles,save_name,angles_ref)

#dihedral
psi_indices, phi_indices = [3, 4, 6, 8], [1, 3, 4, 6]

# phi_angle2 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions2,
#                                                     displacement, phi_indices)
# psi_angle2 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions2,
#                                                     displacement, psi_indices)
# phi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(positions,
#                                                     displacement, phi_indices)
# psi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(positions,
#                                                     displacement, psi_indices)

phi_angle2 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions2,
                                                    displacement2, phi_indices)
psi_angle2 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions2,
                                                    displacement2, psi_indices)
phi_angle3 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions3,
                                                    displacement3, phi_indices)
psi_angle3 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions3,
                                                    displacement3, psi_indices)
phi_angle4 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions4,
                                                    displacement4, phi_indices)
psi_angle4 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions4,
                                                    displacement4, psi_indices)
phi_angle5 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions5,
                                                    displacement5, phi_indices)
psi_angle5 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions5,
                                                    displacement5, psi_indices)

# phi_angle3 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions3,
#                                                     displacement, phi_indices)
# psi_angle3 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions3,
#                                                     displacement, psi_indices)
# phi_angle4 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions4,
#                                                     displacement, phi_indices)
# psi_angle4 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions4,
#                                                     displacement, psi_indices)
# phi_angle5 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions5,
#                                                     displacement, phi_indices)
# psi_angle5 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions5,
#                                                     displacement, psi_indices)
# phi_angle6 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions6,
#                                                     displacement, phi_indices)
# psi_angle6 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions6,
#                                                     displacement, psi_indices)
# phi_angle7 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions7,
#                                                    displacement, phi_indices)
# psi_angle7 = vmap(aa.one_dihedral_displacement, (0,None,None))(positions7,
#                                                     displacement, psi_indices)

phi_angle_ref = vmap(aa.one_dihedral_displacement, (0,None,None))(
                                    positions_ref, displacement, phi_indices)
psi_angle_ref = vmap(aa.one_dihedral_displacement, (0,None,None))(
                                    positions_ref, displacement, psi_indices)

dihedral_angles2 = jnp.stack((phi_angle2,psi_angle2),axis=1)
print(dihedral_angles2.shape)
# dihedral_angles3 = jnp.stack((phi_angle3,psi_angle3),axis=1)
# print(dihedral_angles3.shape)
# dihedral_angles4 = jnp.stack((phi_angle4,psi_angle4),axis=1)
# print(dihedral_angles4.shape)
# dihedral_angles5 = jnp.stack((phi_angle5,psi_angle5),axis=1)
# print(dihedral_angles5.shape)
# dihedral_angles6 = jnp.stack((phi_angle6,psi_angle6),axis=1)
# print(dihedral_angles6.shape)
# dihedral_angles7 = jnp.stack((phi_angle7,psi_angle7),axis=1)
# print(dihedral_angles7.shape)

dihedral_ref = jnp.stack((phi_angle_ref,psi_angle_ref),axis=1)
print(dihedral_ref.shape)

phi_all = jnp.stack((phi_angle_ref,phi_angle2,phi_angle3,phi_angle4,
                            phi_angle5),axis=0)
print(phi_all.shape)
psi_all = jnp.stack((psi_angle_ref,psi_angle2,psi_angle3,psi_angle4,
                            psi_angle5),axis=0)
print(psi_all.shape)
print(len(psi_all))
# onp.save('alanine_dipeptide/confs/phi_angles_s100ns',phi_all)
# onp.save('alanine_dipeptide/confs/psi_angles_s100ns',psi_all)
# if saved_forward_path is not None:
#     # forward_positions = jnp.concatenate((init_ref_position,init_sim_position),axis=0)
#     forward_positions = jnp.stack((init_ref_position,init_sim_position),axis=0)
#     print('forward position', forward_positions.shape)
#     # forward_phi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(
#     #                                 forward_positions, displacement, phi_indices)
#     # forward_psi_angle = vmap(aa.one_dihedral_displacement, (0,None,None))(
#     #                                     forward_positions, displacement, psi_indices)
#     # forward_dihedral_angles = jnp.stack((forward_phi_angle,forward_psi_angle),axis=1)

#     forward_phi_angle = vmap(vmap(aa.one_dihedral_displacement, (0,None,None)),
#                                                 (0,None,None))(forward_positions,
#                                                      displacement, phi_indices)
#     forward_psi_angle = vmap(vmap(aa.one_dihedral_displacement, (0,None,None)),
#                                                 (0,None,None))(forward_positions,
#                                                      displacement, psi_indices)
#     forward_dihedral_angles = jnp.stack((forward_phi_angle,forward_psi_angle),axis=2)
#     print('forward sim start points',forward_dihedral_angles.shape)
#     if plots:
#         visualization.plot_histogram_dihedral(dihedral_angles,save_name+'forward2',
#                                                 init_angles=forward_dihedral_angles)
# dihedral_angles = dihedral_angles2.reshape((400,-1,2))
# print(dihedral_angles2.shape)
# dihedral_angles = dihedral_angles2[:500000]
# dihedral_angles = dihedral_angles2.reshape((split,-1,2))
# print(dihedral_angles.shape)

if plots:
    # from pathlib import Path
    # Path(f'plots/postprocessing/{folder_name}').mkdir(parents=True, exist_ok=True)
    # for i in range(split):
        # visualization.plot_histogram_dihedral(dihedral_angles[i,:],
        #                 save_name+f'_{i}',folder=folder_name)
        # visualization.plot_histogram_free_energy(dihedral_angles[i,:],
        #                 save_name+f'_{i}',folder=folder_name)
        # visualization.plot_histogram_free_energy(dihedral_angles[i,:],save_name+f'_{i}')
        # visualization.plot_histogram_dihedral(dihedral_angles[i,:],save_name+f'_{i}')
        # visualization.plot_1D_free_energy(dihedral_angles[i,:,0],dihedral_ref[:,0],
        #                                     'phi_'+save_name+f'_{i}' ,folder=folder_name)
        # visualization.plot_1D_free_energy(dihedral_angles[i,:,1],dihedral_ref[:,1],
        #    'psi_'+save_name+f'_{i}', color='tab:blue',xlabel='$\psi$',folder=folder_name)
    
    # visualization.plot_free_energy(dihedral_angles2,'test_1d')
    # visualization.plot_histogram_dihedral(dihedral_angles2,save_name+'_2')#,folder=folder_name)
    # visualization.plot_histogram_dihedral(dihedral_angles3,save_name+'_3')#,folder=folder_name)
    # visualization.plot_histogram_dihedral(dihedral_angles4,save_name+'_4')#,folder=folder_name)
    # visualization.plot_histogram_dihedral(dihedral_angles5,save_name+'_5')#,folder=folder_name)
    # visualization.plot_histogram_free_energy(dihedral_angles2,save_name+'_2')#,folder=folder_name)
    # visualization.plot_histogram_free_energy(dihedral_angles3,save_name+'_3')
    # visualization.plot_histogram_free_energy(dihedral_angles4,save_name+'_4')
    # visualization.plot_histogram_free_energy(dihedral_angles5,save_name+'_5')
    # visualization.plot_histogram_free_energy(dihedral_angles6,save_name+'_6')
    # visualization.plot_histogram_free_energy(dihedral_angles7,save_name+'_7')
    # visualization.plot_histogram_free_energy(dihedral_angles2,save_name)
    # visualization.plot_histogram_dihedral(dihedral_angles,save_name,dihedral_ref)
    # visualization.plot_dihedral_diff(dihedral_angles,save_name,dihedral_ref)
    # visualization.plot_scatter_dihedral(dihedral_angles,save_name)
    # visualization.plot_1D_free_energy(dihedral_angles2[:,0],dihedral_ref[:,0],'phi_'+save_name,
    #                                                                            folder=folder_name2)
    # visualization.plot_1D_free_energy(dihedral_angles2[:,1],dihedral_ref[:,1],'psi_'+save_name,
    #                                           color='tab:blue',xlabel='$\psi$',folder=folder_name2)
    phi_angles = [phi_angle,phi_angle2]
    psi_angles = [psi_angle,psi_angle2]
    # phi_angles = [phi_angle2,phi_angle3,phi_angle4,phi_angle5]
    # psi_angles = [psi_angle2,psi_angle3,psi_angle4,psi_angle5]
    # phi_angles = [phi_angle2,phi_angle3,phi_angle4,phi_angle5,phi_angle6,phi_angle7]
    # psi_angles = [psi_angle2,psi_angle3,psi_angle4,psi_angle5,psi_angle6,psi_angle7]
    labels = ['Relative Entropy','Force Matching']
    # labels = ['2','3','4','5']
    visualization.plot_compare_1d_free_energy(phi_angles, dihedral_ref[:, 0], 'phi_' + save_name, labels)
    visualization.plot_compare_1d_free_energy(psi_angles, dihedral_ref[:, 1], 'psi_' + save_name, labels,
                                              xlabel='$\psi$')
# if init_positions is not None:
#     init_phi_angle = vmap(vmap(aa.one_dihedral_displacement, (0,None,None)),
#                                                 (0,None,None))(init_positions,
#                                                      displacement, phi_indices)
#     init_psi_angle = vmap(vmap(aa.one_dihedral_displacement, (0,None,None)),
#                                                 (0,None,None))(init_positions,
#                                                      displacement, psi_indices)

#     init_dihedral_angles = jnp.stack((init_phi_angle,init_psi_angle),axis=2)
#     print(init_dihedral_angles.shape)

#     if plots:
#         visualization.visualize_dihedral_series(init_dihedral_angles,save_name,
#                                                                 dihedral_ref)
################################################################################
# Criteria
dihedral_ref = jnp.deg2rad(dihedral_ref)
dihedral_angles = jnp.deg2rad(dihedral_angles) #convert to rad
# dihedral_angles3 = jnp.deg2rad(dihedral_angles[1,:])
# dihedral_angles4 = jnp.deg2rad(dihedral_angles[2,:])
# dihedral_angles5 = jnp.deg2rad(dihedral_angles[3,:])
# print(dihedral_angles.shape)
# print(dihedral_angles3.shape)
# di1 = dihedral_ref[:400000]
# di2 = dihedral_ref[400000:]
# print(di1.shape, di2.shape)
nbins = 60

h_ref, y1, y2  = onp.histogram2d(dihedral_ref[:,0],dihedral_ref[:,1], bins=nbins, density=True)

mse = []
js = []
kl_scipy = []
kl_entropy = []
# js_scipy = []
# js_e = []

# from scipy.spatial import distance
# from scipy import special

for j in range(split):
    h, _, _  = onp.histogram2d(dihedral_angles[j,:,0],dihedral_angles[j,:,1], bins=nbins, density=True)
    # print(h2.shape)
    # kl_entropy.append(special.rel_entr(h_ref,h2))
    # kl_scipy.append(special.kl_div(h_ref,h2))
    # js_scipy.append(distance.jensenshannon(h_ref,h2))
    mse.append(aa.MSE_energy(h_ref,h,kbT))
    # js.append(aa.JS(h_ref,h2))
    # js_e.append(aa.JS(energy_ref,h_energy))


# h_1, _, _  = onp.histogram2d(dihedral_angles[j,:,0],dihedral_angles[j,:,1], bins=50, density=True)
# h_2, _, _  = onp.histogram2d(dihedral_angles[j,:,0],dihedral_angles[j,:,1], bins=50, density=True)
# h_3, _, _  = onp.histogram2d(dihedral_angles[j,:,0],dihedral_angles[j,:,1], bins=50, densisty=True)
# h2, _, _  = onp.histogram2d(dihedral_angles[j,:,0],dihedral_angles[j,:,1], bins=50, density=True)
# h3, _, _  = onp.histogram2d(dihedral_angles3[:,0],dihedral_angles3[:,1], bins=60, density=True)
# h4, _, _  = onp.histogram2d(dihedral_angles4[:,0],dihedral_angles4[:,1], bins=60, density=True)
# h5, _, _  = onp.histogram2d(dihedral_angles5[:,0],dihedral_angles5[:,1], bins=60, density=True)

# h_2, _, _  = onp.histogram2d(di2[:,0],di2[:,1], bins=60, density=True)
# h_1, _, _  = onp.histogram2d(di1[:,0],di1[:,1], bins=60, density=True)

print('each 10ns')
print('mse, (mean/std)')
print(onp.mean(mse),onp.std(mse))
print(onp.min(mse),onp.max(mse))

# print('js, (mean/std)')
# print(onp.mean(js),onp.std(js))
# print(onp.min(js),onp.max(js))


# dihedral_angles3 = dihedral_angles2.reshape((4,-1,2))
# dihedral_angles3 = jnp.deg2rad(dihedral_angles3)
# mse3 = []
# js3 = []

# for k in range(4):
#     h3, _, _  = onp.histogram2d(dihedral_angles3[k,:,0],dihedral_angles3[k,:,1], bins=nbins, density=True)
#     mse3.append(aa.MSE_energy(h_ref,h3,kbT))
#     js3.append(aa.JS(h_ref,h3))


# print('stacked as 100ns')
# print('mse, (mean/std)')
# print(onp.mean(mse3),onp.std(mse3))
# print(onp.min(mse3),onp.max(mse3))

# print('js, (mean/std)')
# print(onp.mean(js3),onp.std(js3))
# print(onp.min(js3),onp.max(js3))

dihedral_angles2 = jnp.deg2rad(dihedral_angles2)
# dihedral_angles3 = jnp.deg2rad(dihedral_angles3)
# dihedral_angles4 = jnp.deg2rad(dihedral_angles4)
# dihedral_angles5 = jnp.deg2rad(dihedral_angles5)
# dihedral_angles6 = jnp.deg2rad(dihedral_angles6)
# dihedral_angles7 = jnp.deg2rad(dihedral_angles7)
h2, _, _  = onp.histogram2d(dihedral_angles2[:,0],dihedral_angles2[:,1], bins=nbins, density=True)
# h3, _, _  = onp.histogram2d(dihedral_angles3[:,0],dihedral_angles3[:,1], bins=nbins, density=True)
# h4, _, _  = onp.histogram2d(dihedral_angles4[:,0],dihedral_angles4[:,1], bins=nbins, density=True)
# h5, _, _  = onp.histogram2d(dihedral_angles5[:,0],dihedral_angles5[:,1], bins=nbins, density=True)
# h6, _, _  = onp.histogram2d(dihedral_angles6[:,0],dihedral_angles6[:,1], bins=nbins, density=True)
# h7, _, _  = onp.histogram2d(dihedral_angles7[:,0],dihedral_angles7[:,1], bins=nbins, density=True)



print('all (mse,js)')
mse2 = aa.MSE_energy(h_ref,h2,kbT)
print(mse2)
# mse3 = aa.MSE_energy(h_ref,h3,kbT)
# mse4 = aa.MSE_energy(h_ref,h4,kbT)
# mse5 = aa.MSE_energy(h_ref,h5,kbT)
# mse6 = aa.MSE_energy(h_ref,h6,kbT)
# mse7 = aa.MSE_energy(h_ref,h7,kbT)

# mse_all = jnp.array([mse2,mse3,mse4,mse5,mse6,mse7])
# print(mse_all)
# print(jnp.mean(mse_all))
# print(jnp.std(mse_all))
# print(aa.JS(h_ref,h))


# import matplotlib.pyplot as plt
# def plot_mean_histogram(angles,saveas,kbT=2.49435321,degrees=True,folder=''):
#     '''Plot and save 2D histogram for alanine dipeptide free energies
#     from the dihedral angles.'''
#     cmap = plt.get_cmap('magma')

#     if degrees:
#         angles = jnp.deg2rad(angles)

#     h, x_edges, y_edges  = jnp.histogram2d(angles[:,0],angles[:,1],
#                                                     bins = 60, density=True)

#     h = jnp.log(h)*-kbT/4.184
#     x, y = onp.meshgrid(x_edges, y_edges)

#     plt.figure()
#     plt.pcolormesh(x, y, h.T,cmap=cmap)
#     # axs.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
#     # axs.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
#     cbar = plt.colorbar()
#     cbar.set_label('Free Energy (kcal/mol)')
#     plt.xlabel('$\phi$')
#     plt.ylabel('$\psi$')
#     plt.savefig(f'plots/postprocessing/{folder}histogram_mean_{saveas}.png')
#     plt.show()
#     plt.close('all')
#     return
# print('js energy')
# print(onp.mean(js_e),onp.std(js_e))
# print(onp.min(js_e),onp.max(js_e))
# print('js scipy')
# print(onp.mean(js_scipy),onp.std(js_scipy))
# print(onp.min(js_scipy),onp.max(js_scipy))
# # print('kl scipy')
# # print(onp.mean(kl_scipy),onp.std(kl_scipy))
# # print(onp.min(kl_scipy),onp.max(kl_scipy))
# # print('kl entropy')
# print(onp.mean(kl_entropy),onp.std(kl_entropy))
# print(onp.min(kl_entropy),onp.max(kl_entropy))
# print(aa.MSE_energy(h_ref,h2,kbT))


# print('vs 3')
# print(aa.MSE_energy(h_ref,h3,kbT))
# print(aa.JS(h_ref,h3))

# print('vs 4')
# print(aa.MSE_energy(h_ref,h4,kbT))
# print(aa.JS(h_ref,h4))

# print('vs 5')
# print(aa.MSE_energy(h_ref,h5,kbT))
# print(aa.JS(h_ref,h5))


# print('vs100k')
# print(aa.MSE_energy(h_2,h,kbT))
# print(aa.JS(h_2,h))

# print('vs400k')
# print(aa.MSE_energy(h_1,h,kbT))
# print(aa.JS(h_1,h))
################################################################################