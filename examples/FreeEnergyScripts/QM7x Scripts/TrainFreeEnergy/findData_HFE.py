#import pyanitools as pya
import pandas as pd
import re
import pickle
import h5py
from rdkit import Chem




def get_FreeSolv_data():
    """Get data from FreeSolv pickle file"""
    # Extract primary data from original database pickle file.
    original_database_filename = '../FreeSolvDB/database.pickle'
    pd_pickle = pd.read_pickle(original_database_filename)
   # breakpoint()
    return pd_pickle


def get_relevant_molecules(dic_mol):
    """Passing the Freesolv DB file returned by get_FreeSolv_data, this
    function returns molecules which only contain CcNnOo and have max 8 heavy atoms.
    Maybe also take values with d_expt < 0.6?"""
    red_mols = dict()
    for mol_name in dic_mol.keys():
       # breakpoint()
        if not re.search("[AaBbDdEeFfGgIiJjKkLlMmPpQqRrSsTtUuVvWwXxYyZz]", dic_mol[mol_name]['smiles']):
            # Add length of moles
            x = re.findall('[CcNnOo]', dic_mol[mol_name]['smiles'])
            # y = re.findall('Cl', dic_mol[mol_name]['smiles'])
            # z = re.findall('C(?!l)', dic_mol[mol_name]['smiles'])
            # if len(x + y + z) < 8:
            red_mols[mol_name] = dic_mol[mol_name]
            red_mols[mol_name]['length'] = len(x)

    return red_mols

def get_relevant_molecules_2(dic_mol):
    """Passing the Freesolv DB file returned by get_FreeSolv_data, this
    function returns molecules which only contain CcNnOo and have max 8 heavy atoms.
    Maybe also take values with d_expt < 0.6?"""
    red_mols = dict()
    for mol_name in dic_mol.keys():
       if not re.search("[AaBbDdEeFfGgIiJjKkMmPpQqRrTtUuVvWwXxYyZz]", dic_mol[mol_name]['smiles']):
            # Add length of moles
            x = re.findall('[cNnOo]', dic_mol[mol_name]['smiles'])
            y = re.findall('Cl', dic_mol[mol_name]['smiles'])
            z = re.findall('C(?!l)', dic_mol[mol_name]['smiles'])
            # if len(x + y + z) < 8:
            red_mols[mol_name] = dic_mol[mol_name]
            red_mols[mol_name]['length'] = len(x + y + z)

    return red_mols


# freesolv_dict = get_FreeSolv_data()
#
# ## Selecting molecules of N-H-O-C composition
# relevant_molecules_dict = get_relevant_molecules(freesolv_dict)
#
# ## Forming dictionary of relevant molecules with experimental Hydration free energy values + errors
# solvation_dictionary = {}
# for mol in relevant_molecules_dict.keys():
#     solvation_dictionary[relevant_molecules_dict[mol]['smiles']] = {'expSolvFreeEnergy': relevant_molecules_dict[mol]['expt'], # free energy units in kcal/mol
#                                                        'expUncertainty': relevant_molecules_dict[mol]['d_expt']}
# ## Creating list of SMILES
# smile_list = []
# numbers = []
# for smile in solvation_dictionary.keys():
#     smile_list.append(smile)
#     mol = Chem.MolFromSmiles(smile)
#     mol = Chem.AddHs(mol)
#     number = mol.GetNumAtoms()
#     numbers.append(number)
# print(len(smile_list))
# print(max(numbers))


def create_hdf5_dataset_matching_ANI1_and_FreeSolv(freeSolv_and_ANI_smiles):
    """Function to create hdf5 dataset for matching ANI1 and FreeSolv molecules.
    For each entry of matched values between ANI-1 and FreeSolv DB get the std_smiles
    values, and iterate over respective ANI DB, find matching entry and save SMILES, coordinates
    and energies in hdf 5 file."""
    # Remove database if exists
    with h5py.File('solvationMolecules_hdf5.h5', 'w') as hdf:
        pass

    count_matches = 1
    for i in freeSolv_and_ANI_smiles.keys():
        # Get length of molecule
        len_mol = freeSolv_and_ANI_smiles[i]
        # Get ANI-1 dictionary of this molecule
        name_pickle_file = 'ANI_DB_{}'.format(len_mol)
        with open(name_pickle_file, 'rb') as handle:
            ani_dic_temp = pickle.load(handle)
        # Get string of ANI-1 DB -> ANI-1 smiles saved as e.g. 'H','[','O',']',...
        ani1_str = ani_dic_temp[i]['str_smiles']
        # Get hdf5 with molecules of respecive length
        hdf5file_temp = '../ANI-1_release/ani_gdb_s0{}.h5'.format(len_mol)
        # Construct the data loader class
        adl = pya.anidataloader(hdf5file_temp)
        # Iterate over all entries in temp hdf5 file, find ani1_str and write out all relevant information
        for data in adl:
            if data['smiles'] == ani1_str:
                print('Found respective value for {} match'.format(count_matches))
                count_matches += 1
                # Write out smiles, coordinates, energies, and solvation free energy to hdf 5 file
                breakpoint()
                relevant_information = {'smiles': i, 'coordinates': data['coordinates'],
                                        'energies': data['energies'],
                                        'expSolvFreeEnergy': solvation_dictionary[i]['expSolvFreeEnergy'],
                                        'expSolvUncertainty': solvation_dictionary[i]['expUncertainty']}
                # Write out to hdf5 file
                name_dataset = "molecule_{}".format(i)
                with h5py.File('solvationMolecules_hdf5.h5', 'a') as hdf:
                    G1 = hdf.create_group(i)
                    G1.create_dataset('Coordinates', data=relevant_information['coordinates'])
                    G1.create_dataset('Energies', data=relevant_information['energies'])
                    G1.create_dataset('ExpSolvFreeEnergy', data=relevant_information['expSolvFreeEnergy'])
                    G1.create_dataset('ExpSolvUncertainty', data=relevant_information['expSolvUncertainty'])
                    print('Done with molecule {}'.format(count_matches))


if __name__ == '__main__':
    # 1. Go through FreeSolv data, save smiles and corresponding hydration energy.
    data_base = get_FreeSolv_data()
    reduced_db = get_relevant_molecules(data_base)

    # Create dictionary with {smiles: {exp. solvation energy: number, uncertainty solv. free energy: value}}
    # to use when creating hdf5 database.
    solvation_dictionary = {}
    for mol in reduced_db.keys():
        solvation_dictionary[reduced_db[mol]['smiles']] = {'expSolvFreeEnergy': reduced_db[mol]['expt'], 'expUncertainty': reduced_db[mol]['d_expt']}
        #print('{} has {} expSolvFreeEnergy and {} expUncertainty.'.format(reduced_db[mol]['smiles'], reduced_db[mol]['expt'], reduced_db[mol]['d_expt']))

    # List with all smiles that are in FreeSolv DB and ANI-1 DB
    freeSolv_and_ANI_smiles = {}
    # 2. Go through adjust smiles
    for mol in reduced_db.keys():
        len_molecule = reduced_db[mol]['length']
        if len_molecule <= 8:
            # check if value in ANI-1 database from dictionary with all ANI-1 smiles (before created)
            name_pickle_file = 'ANI_DB_{}'.format(len_molecule)
            with open(name_pickle_file, 'rb') as handle:
                ani_dic = pickle.load(handle)

        # Create a list of smiles and check if found in ANI-1 DB
            ani_list = list(ani_dic)
            if (ani_list.count(reduced_db[mol]['smiles']) == 1):
                print('Found a True value - happy up :)')
                print(reduced_db[mol]['smiles'])
                freeSolv_and_ANI_smiles[reduced_db[mol]['smiles']] = len_molecule
                #freeSolv_and_ANI_smiles.append(reduced_db[mol]['smiles'])

    # Create hdf5 dataset
    create_hdf5_dataset_matching_ANI1_and_FreeSolv(freeSolv_and_ANI_smiles)


    print('Totally found {} coincident smiles entries in FreeSolv DB and ANI-1 DB'.format(len(freeSolv_and_ANI_smiles)))

    # 3. Find given smiles in ANI-1 db and create a np array with the following structure:
    # [smile, FreeSolv hydration energy, Energy of ANI-1 DB, Coordinates of ANI-1 conformations.]
