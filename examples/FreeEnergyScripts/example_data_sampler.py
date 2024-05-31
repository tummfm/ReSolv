import pyanitools as pya
import pickle


def create_ANI_pickle_db():
    """For each ANI-1 molecule with a length of 1-8 heavy atoms, write a pickle file that contains the entries
    smiles_with_h and str_smiles."""
    # Set the HDF5 file containing the data
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        length_molecule = i
        hdf5file = 'ANI1DB/ani_gdb_s0{}.h5'.format(length_molecule)

        # Construct the data loader class
        adl = pya.anidataloader(hdf5file)

        # Dictionary with new smiles values
        new_smiles = dict()

        # Print the species of the data set one by one
        for data in adl:
            # Extract the data
            P = data['path']
            X = data['coordinates']
            E = data['energies']
            S = data['species']
            sm = data['smiles']
            ch_smile_old = ''.join(data['smiles'])
            ch_smile = ch_smile_old.replace('([H])', '')
            ch_smile = ch_smile.replace('[H]', '')
            new_smiles[ch_smile] = {'smiles_with_h': ch_smile_old, 'str_smiles': sm}

        # Save as pickle - Need to be done once for each dataset
        name_pickle_file = 'ANI1DB/ANI_DB_{}'.format(length_molecule)
        with open(name_pickle_file, 'wb') as file:
            pickle.dump(new_smiles, file)

        print('Done')

        # Closes the H5 data file
        adl.cleanup()

if __name__ == '__main__':
    create_ANI_pickle_db()