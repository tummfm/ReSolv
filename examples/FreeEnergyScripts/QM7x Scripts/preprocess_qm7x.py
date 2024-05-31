import os

visible_device = ""
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import h5py
from sys import stdout
import numpy as onp

from chemtrain.sparse_graph import pad_forces_positions_species


## function to exclude duplicates from QM7-X dataset
def removing_duplicates(IDs):
    DupMols = []
    for line in open('DupMols.dat', 'r'):
        DupMols.append(line.rstrip('\n'))

    for IDconf in IDs:
        if IDconf in DupMols:
            IDs.remove(IDconf)
            stmp = IDconf[:-3]
            for ii in range(1, 101):
                IDs.remove(stmp + 'd' + str(ii))

    return IDs
def get_atoms_buffer_property_buffer():
    # atom energies
    EPBE0_atom = {6: -1027.592489146, 17: -12516.444619523, 1: -13.641404161,
                  7: -1484.274819088, 8: -2039.734879322, 16: -10828.707468187}
    atom_positions = []
    atom_numbers = []
    atom_energies = []
    atom_forces = []
    atom_molecule = []
    stdout.write('\n')

    ## for all sets of molecules (representing individual files):
    set_ids = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000']
    # set_ids = ['1000']
    for setid in set_ids:
        ## load HDF5 file
        fMOL = h5py.File('QM7x_DB/' + setid + '.hdf5', 'r')

        ## get IDs of HDF5 files and loop through
        mol_ids = list(fMOL.keys())
        for molid in mol_ids:
            stdout.write('Current molecule: ' + molid + '\n  Conformations:')

            ## get IDs of individual configurations/conformations of molecule
            conf_ids = list(fMOL[molid].keys())

            ## use this option if you want to exclude duplicates
            #        conf_ids = removing_duplicates(conf_ids)

            for confid in conf_ids:
                ## get atomic positions and numbers and add to molecules buffer
                xyz = onp.array(fMOL[molid][confid]['atXYZ'])
                Z = onp.array(fMOL[molid][confid]['atNUM'])
                # atoms_buffer.append(Atoms(Z, xyz))
                energy_tot = float(list(fMOL[molid][confid]['ePBE0+MBD'])[0])
                force = onp.array(list(fMOL[molid][confid]['totFOR']))
                'Geom-m10-i1-c1-opt'

                atom_positions.append(xyz)
                atom_numbers.append(Z)
                atom_energies.append(energy_tot)
                atom_forces.append(force)
                atom_molecule.append(molid)


                # ## get quantum mechanical properties and add them to properties buffer
                # ## The user decides the properties to save in the DB file (see README.txt)
                # force = list(fMOL[molid][confid]['totFOR'])
                # Eatoms = sum([EPBE0_atom[zi] for zi in Z])
                # Eat = float(list(fMOL[molid][confid]['eAT'])[0])
                # EPBE0 = float(list(fMOL[molid][confid]['ePBE0'])[0])
                # EMBD = float(list(fMOL[molid][confid]['eMBD'])[0])
                # C6 = float(list(fMOL[molid][confid]['mC6'])[0])
                # POL = float(list(fMOL[molid][confid]['mPOL'])[0])
                # HLGAP = float(list(fMOL[molid][confid]['HLgap'])[0])
                # DIP = float(list(fMOL[molid][confid]['DIP'])[0])
                # property_buffer.append({'forces': np.array(force),
                #                         'EPBE0': np.array([EPBE0]),
                #                         'Eat': np.array([Eat]),
                #                         'EMBD': np.array([EMBD]),
                #                         'C6': np.array([C6]),
                #                         'POL': np.array([POL]),
                #                         'HLGAP': np.array([HLGAP]),
                #                         'DIP': np.array([DIP])})

            stdout.write('Molecule ' + molid + ' done.\n')
        print("Done with ID {}".format(setid))

    return atom_positions, atom_numbers, atom_energies, atom_forces, atom_molecule


atom_positions, atom_numbers, atom_energies, atom_forces, atom_molecule = get_atoms_buffer_property_buffer()

# Pad the data, encode data into numpy arrays
padded_pos, padded_forces, padded_species = pad_forces_positions_species(species=atom_numbers,
                                                                         position_data=atom_positions,
                                                                         forces_data=atom_forces)
atom_energies = onp.array(atom_energies)
atom_molecule = onp.array(atom_molecule)
# Save
onp.save("QM7x_DB/atom_positions_QM7x.npy", padded_pos)
onp.save("QM7x_DB/atom_numbers_QM7x.npy", padded_species)
onp.save("QM7x_DB/atom_energies_QM7x.npy", atom_energies)
onp.save("QM7x_DB/atom_forces_QM7x.npy", padded_forces)
onp.save("QM7x_DB/atom_molecule_QM7x.npy", atom_molecule)

# Shuffle the data and save a shuffled set
indices = onp.arange(len(atom_energies))
onp.random.shuffle(indices)
print("First 20 shuffled indices: ", indices[:20])

shuffled_padded_pos = padded_pos[indices]
shuffled_padded_species = padded_species[indices]
shuffled_atom_energies = atom_energies[indices]
shuffled_padded_forces = padded_forces[indices]
shuffled_atom_molecule = atom_molecule[indices]

# Save the shuffled data
onp.save("QM7x_DB/shuffled_atom_positions_QM7x.npy", shuffled_padded_pos)
onp.save("QM7x_DB/shuffled_atom_numbers_QM7x.npy", shuffled_padded_species)
onp.save("QM7x_DB/shuffled_atom_energies_QM7x.npy", shuffled_atom_energies)
onp.save("QM7x_DB/shuffled_atom_forces_QM7x.npy", shuffled_padded_forces)
onp.save("QM7x_DB/shuffled_atom_molecule_QM7x.npy", shuffled_atom_molecule)


print("Done")

