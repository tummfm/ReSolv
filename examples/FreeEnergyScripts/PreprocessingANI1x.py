import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)


import h5py
import numpy as np

from chemtrain.sparse_graph import pad_forces_positions_species

os.environ["CUDA_VISIBLE_DEVICES"] = str(5)

# 1. Function to get ANI1x-dataset. Access positions, species, forces, and energies.
# From ANI1x_dataset git repository
def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d


def get_ani1x_dataset(data_type):
    """Function content copied from example loader in ANI1x_datasets git repository.
    Extracts DFT energies, forces, coordinates and atomic numbers form dataset."""
    ani1_coordinates = []
    ani1_species = []
    ani1_energies = []
    ani1_forces = []


    path_to_h5file = 'ANI1xDB/ani1x-release.h5'
    # path_to_h5file = '/home/sebastien/FreeEnergySolubility/examples/FreeEnergyScripts/ANI1xDB/ani1x-release.h5'

    # List of keys to point to requested data
    if data_type == "ANI-1x":
        data_keys = ['wb97x_dz.energy', 'wb97x_dz.forces']  # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
    # data_keys = ['wb97x_tz.energy','wb97x_tz.forces'] # CHNO portion of the data set used in AIM-Net (https://doi.org/10.1126/sciadv.aav6490)
    elif data_type == "ANI-1ccx":
        data_keys = ['ccsd(t)_cbs.energy'] # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
    # data_keys = ['wb97x_dz.dipoles'] # A subset of this data was used for training the ACA charge model (https://doi.org/10.1021/acs.jpclett.8b01939)
    else:
        print("Neither ANI-1x nor ANI-1ccx chosen. Exit.")
        exit(1)

    # Example for extracting DFT/DZ energies and forces
    counter = 0
    sum_samples = 0
    for data in iter_data_buckets(path_to_h5file, keys=data_keys):
        X = data['coordinates']
        Z = data['atomic_numbers']
        E = data['wb97x_dz.energy']
        F = data['wb97x_dz.forces']
        print("Debug")

        ani1_coordinates += list(X)
        ani1_species += list(np.reshape(list(Z) * X.shape[0], (X.shape[0], Z.shape[0])))
        ani1_energies += list(E)
        ani1_forces += list(F)
        counter += 1
        sum_samples += X.shape[0]
        print("Counter: {}, sum_samples: {}".format(counter, sum_samples))
    return ani1_coordinates, ani1_species, ani1_energies, ani1_forces



# 2. Function to pad the molecules. Pad forces, positions, and species. Also get edge mask.
def pad_molecules(species, position_data, forces_data):
    padded_pos, padded_forces, species_mask = pad_forces_positions_species(species=species, position_data=position_data,
                                                                           forces_data=forces_data)
    return padded_pos, padded_forces, species_mask



if __name__ == '__main__':

    ani1x_coordinates, ani1x_species, ani1x_energies, ani1x_forces = get_ani1x_dataset(data_type="ANI-1x")

    # Arrays are returned as onp arrays
    pad_pos, pad_forces, mask_species = pad_molecules(species=ani1x_species, position_data=ani1x_coordinates,
                                                      forces_data=ani1x_forces)

    # Save padded pos, forces, mask_species, mask_force, mask_edge
    np.save("ANI1xDB/pad_pos.npy", pad_pos)
    np.save("ANI1xDB/pad_forces.npy", pad_forces)
    np.save("ANI1xDB/mask_species.npy", mask_species)
    np.save("ANI1xDB/energies.npy", ani1x_energies)
    print("Debug")