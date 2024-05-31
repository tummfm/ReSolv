import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import numpy as onp
from chemtrain.jax_md_mod import custom_space

################################################
# md_positions = np.load('confs/confs_heavy_100ns.npy')
md_positions = onp.load('confs/confs_FM_100ns_problem_before.npy')
# md_positions = np.load('../../examples/output/alanine_confs/confs_test_alanine_RE_both_prior_100gamma_002_10ns.npy')
################################################
name = 'FM_100ns_problem_before'

n_every = 10 #take every n frames

format = 'gro'

print(md_positions.shape)

print(onp.max(md_positions),onp.min(md_positions))

n_snaps = md_positions.shape[0]
n_atoms = md_positions.shape[1]
print(n_snaps)
print(n_atoms)
# traj = mdtraj.load_trr('MD/100ns/md.trr', top='MD/100ns/water.gro', stride=None, atom_indices=None, frame=True)
# print(traj)
# print(traj.xyz.shape)

# phi = mdtraj.compute_phi(traj, periodic=False, opt=True)
# psi = mdtraj.compute_psi(traj, periodic=False, opt=True)
# print(md_positions[0])

protein = ['CH3','C','O','N','CA','CB','C','O','N','CH3']
residue = ['ACE','ACE','ACE','ALA','ALA','ALA','ALA','ALA','NME','NME']
numbers = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
# box = [1.0, 1.0, 1.0]


box = onp.array([2.71381, 2.71381, 2.71381])
box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)

# displacement, shift = space.periodic_general(box_tensor,
#                                          fractional_coordinates=True)

inv_scale_fn = lambda R: onp.dot(R, box_tensor)

md_positions = inv_scale_fn(md_positions)

timestep = 0.002
printout = 0.1

if format == 'xyz':
    outfile = open(f'data/CG_alanine_{name}.xyz', 'w')
    for i in range(int(n_snaps/n_every)):
        outfile.write('10\n')
        outfile.write('Alanine dipeptide time = %d\n'%i)
        for j in range(n_atoms):
            outfile.write('%s %.15g %.15g %.15g\n'%(protein[j], md_positions[i*n_every,j,0], 
                                                            md_positions[i*n_every,j,1], md_positions[i*n_every,j,2]))
    outfile.close()

elif format == 'gro':
    outfile = open(f'data/CG_alanine_{name}.gro', 'w')
    for i in range(int(n_snaps/n_every)):
        outfile.write('Alanine dipeptide heavy atoms, t= %.5f step= %d\n'%(i*n_every*timestep, i*n_every*printout/timestep))
        outfile.write('%5d\n'%(n_atoms))
        for j in range(n_atoms):
            outfile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'%(numbers[j], residue[j], protein[j], j+1,
                                                md_positions[i*n_every,j,0], md_positions[i*n_every,j,1], md_positions[i*n_every,j,2]))
        outfile.write('%10.5f%10.5f%10.5f\n'%(box[0],box[1],box[2]))
    outfile.close()

else:
    raise ValueError(f'Format {format} not implemented. Should'
                                f' be xyz or gro.')