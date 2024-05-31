import os
visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import h5py


# Load COMP6 - ANI - MD
h5filename = "COMP6-master/COMP6v1/ANI-MD/ani_md_bench.h5"

all_coordinates = []
all_energies = []
all_forces = []
all_species = []
species_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
with h5py.File(h5filename, 'r') as f:
    for data in f.keys():
        for props in f[data]:
            for entries in props:
                coordinates = onp.array(f[data][props]['coordinates'])
                energies = onp.array(f[data][props]['energies'])
                forces = onp.array(f[data][props]['forces'])
                species = f[data][props]['species']
                species = [species_dict[i.decode()] for i in species]
                species = onp.tile(species, len(energies)).reshape(len(energies), -1)

                # Append the entries
                all_coordinates.append(coordinates)
                all_energies.append(energies)
                all_forces.append(forces)
                all_species.append(species)





# Choose setup
from jax_md import space, partition
from jax import value_and_grad, numpy as jnp
from chemtrain import neural_networks
from functools import partial
import pickle

ID = "DimeNet_ID5"
all_size = 100287
num_epochs = 50
path_energy_params = "savedTrainers/120422" + str(ID) + "_Trainer_ANI1x_" + str(all_size) + "Samples_" + str(
    num_epochs) + "epochs_gu2emin7_gf1emin3_LRdecay1emin3_Energies_bestParams.pkl"

r_cut = 0.5
n_species = 10

check_num = 0
positions_check = all_coordinates[check_num]
energy_check = all_energies[check_num]
species_check = all_species[check_num]
# Å to nm
box = jnp.eye(3) * 100
positions_check *= 0.1
for i, pos in enumerate(positions_check):
    positions_check[i] += onp.array([50., 50., 50.])

displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)

neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                      dr_threshold=0.,
                                      capacity_multiplier=0.,
                                      fractional_coordinates=True,
                                      disable_cell_list=True)


_, GNN_energy = neural_networks.dimenetpp_neighborlist_ANI1x(
    displacement, r_cut, n_species, positions_test=None, neighbor_test=None,
    max_edges=None, max_angles=None, kbt_dependent=False, embed_size=64)


def energy_fn_template(energy_params):
    # Took out species in partial() to force that species is passed, as this varies.
    gnn_energy = partial(GNN_energy, energy_params)

    def energy(R, neighbor, species, **dynamic_kwargs):
        return gnn_energy(positions=R, neighbor=neighbor, species=species, **dynamic_kwargs)

    return energy


with open(path_energy_params, 'rb') as pickle_file:
    loaded_params = pickle.load(pickle_file)

for i in range(len(positions_check)):
    # Iterate over the data and make predictions
    nbrs_init = neighbor_fn.allocate(positions_check[i], extra_capacity=0)
    energy_fn_init = energy_fn_template(energy_params=loaded_params)
    pred_energies, neg_forces = value_and_grad(energy_fn_init)(positions_check[i], neighbor=nbrs_init,
                                                               species=species_check[i])
    print("Predicted energy: ", pred_energies)
    print("True energy: ", energy_check[i])

