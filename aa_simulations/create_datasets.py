"""Loads LAMMPS and GROMACS files and saves the dataset (with or without CG
mapping), currently hardcoded on water and Alanine.
"""
import os
import sys

import aa_simulations.io

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = None
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform, visible_device)

import numpy as onp
from chemtrain.jax_md_mod import io
from aa_util import moleculewise_com
from jax import vmap
from jax_md import space
import aa_util as aa

################################################################################
# Water
#Units
mass1 = 15.9994 #g/mol
mass2 = 1.00794 #g/mol
Molemass = 18.01528 #g/mol
atm_to_kj_per_mol_nm3 = 101325*6.0221367e-7 #atm in Pa, Pa in kj/mol-nm3
#use same ordering as in lammps. Needs to be in shape (Natoms/mol x 1)
masses = onp.array([[mass1],[mass2],[mass2],[4]])

dumpfile = 'water/data/trj/tip4p_1nsfinal.trj'
logfile = 'water/data/logs/log_1nsfinal.lammps'
# dumpfile = 'water/data/trj/tip4p_1nsfinal.trj'
# logfile = 'water/data/logs/log_1nsfinal.lammps'
referencefile = 'water/data/RDF/wat_10nsfinal.rdf'
# dumpfile = 'water/data/trj/LJfluid_27.trj'
# logfile = 'water/data/logs/log_lj_27.lammps'
rdffile = 'water/data/RDF/COM_10k_rdf.csv'


saveas = 'test'
com = True # True: get COM position, forces and length, False: get all atom position/forces

ref_rdfs = onp.loadtxt(referencefile, skiprows=4, usecols=(1,2)) #for lammps .rdf files
ref_rdfs[:,0] = ref_rdfs[:,0]/10 #convert A->nm
print(ref_rdfs.shape)
rdf = onp.loadtxt(rdffile) #for .csv RDF files
print(rdf.shape)
x = onp.linspace(0, 1, num=rdf.shape[0])
print(x.shape)
rdfs = onp.column_stack((x,rdf))

# aa.plot_rdf(rdfs,saveas,ref_rdfs)

print('Finsished plotting rdf')
    
positions, forces, length = aa_simulations.io.read_dump(dumpfile)

if com:
    print('Creating COM arrays for',saveas,'timesteps')
    n_apm = len(masses) #Natoms/molecule
    n_atoms = positions.shape[1]
    n_steps = positions.shape[0]
    box = onp.array([length, length, length])
    displacement, shift = space.periodic(box)

    if onp.mod(n_atoms,n_apm) != 0: #Does only check if Nmol makes sense. Does not check if lammps Natoms/mol = provided Nmasses
        raise ValueError('Number of atoms/mol'+str(n_atoms)+'does not match number of provided masses'+str(n_apm))

    #reshape into (Nstep x Nmol x Natoms/molecule x 3) array with (Number molec,positions per molecule (this case 3 per molecule),xyz)
    positions = vmap(vmap(moleculewise_com, (0,None,None)), (0,None,None))(onp.reshape(positions,(n_steps,-1,n_apm,3)),masses,displacement)
    forces = onp.reshape(forces,(n_steps,-1,n_apm,3)).sum(2) #sum forces of each molecule

    # onp.save(f'confs/conf_COM_{saveas}', positions)
    # onp.save(f'confs/forces_COM_{saveas}', forces)
    # onp.save(f'confs/length_COM_{saveas}', length)
else:
    print('Creating atom arrays for',saveas,'timesteps')

    # onp.save(f'confs/conf_atoms_{saveas}', positions)
    # onp.save(f'confs/forces_atoms_{saveas}', forces)
    # # pos = positions[:,::3,:].copy() #for oxgens only
    # # onp.save(f'confs/conf_O_{saveas}', pos)
    # onp.save(f'confs/length_atoms_{saveas}', length)

print('Finsished creating forces, positions')

key_dict = {'Step': 1, 'c_presstensor': 6, 'c_stresstensor': 6}
result_dict = aa_simulations.io.read_log(logfile, key_dict)

#unit conversion
Pv = result_dict['c_presstensor']*atm_to_kj_per_mol_nm3
Sv = result_dict['c_stresstensor']*atm_to_kj_per_mol_nm3

# onp.save(f'confs/virial_pressure_{saveas}', Pv)
# onp.save(f'confs/virial_stress_{saveas}', Sv)

print('Finsished creating virial pressure, stress')

################################################################################
# Alanine Dipeptide
file_trr = 'alanine_dipeptide/data/1000ns/heavyMD.trr'
file_gro = 'alanine_dipeptide/data/1000ns/heavy_first.gro'
#Create Positions, Forces
positions_ala, forces_ala = aa_simulations.io.trr_to_numpy(file_gro, file_trr, force=True)

onp.save('confs/confs_heavy_1000ns',positions_ala)
onp.save('confs/forces_heavy_1000ns',forces_ala)
print('Positions:',positions_ala.shape)
print('Forces:',forces_ala.shape)
print(forces_ala[0,0,0],positions_ala[0,0,0])
#Loading structure
# box_ala, R_ala, masses_ala, species_ala, bonds_ala = io.load_box(file_gro)