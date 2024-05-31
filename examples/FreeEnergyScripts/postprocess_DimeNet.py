import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import pickle
from jax_md import space, partition
from jax import numpy as jnp, lax, value_and_grad
from functools import partial


from chemtrain import neural_networks
from chemtrain.jax_md_mod import custom_space

if __name__ == '__main__':
    compute_RMSE = True
    if compute_RMSE:
        # path_energy_params = "savedTrainers/060422_Trainer_ANI1x_100287Samples_50epochs_gu2emin7_gf1emin3_LRdecay1emin3_EnergiesAndForces_bestParams.pkl"

        # Choose setup
        ID = "DimeNet_ID5"
        all_size = 100287
        num_epochs = 50
        path_energy_params = "savedTrainers/120422" + str(ID) + "_Trainer_ANI1x_" + str(all_size) + "Samples_" + str(
            num_epochs) + "epochs_gu2emin7_gf1emin3_LRdecay1emin3_Energies_bestParams.pkl"
        # path_energy_params = "savedTrainers/130422" + str(ID) + "_Trainer_ANI1x_" + str(all_size) + "Samples_" + str(
        #     num_epochs) + "epochs_gu2emin7_gf1emin3_LRdecay1emin3_Energies_lastParams.pkl"
        load_dataset_name = 'ANI1xDB/'

        pad_pos = onp.load(load_dataset_name + 'pad_pos.npy')[:all_size]
        pad_species = onp.load(load_dataset_name + 'mask_species.npy')[:all_size]

        # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
        pad_species = onp.array(pad_species, dtype='int32')

        # Å to nm
        pad_pos *= 0.1
        for i, pos in enumerate(pad_pos):
            pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])

        for i, pos in enumerate(pad_pos):
            x_spacing = 0
            for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                x_spacing += 0.6
                pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing

        box = jnp.eye(3) * 100

        scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
        pad_pos = lax.map(scale_fn, pad_pos)
        # Turn all arrays to jax.numpy format
        pad_species = jnp.array(pad_species)
        pad_pos = jnp.array(pad_pos)
        init_pos = pad_pos[0]   # Important for initialize neighborlist
        init_species = pad_species[0]


        # Load energy_fn to make predictions
        r_cut = 0.5
        dr_thresh = 0.05
        neighbor_capacity_multiple = 2.7  # Hard coded for ANI1-x dataset.
        displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)


        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.7,
                                              fractional_coordinates=True,
                                              disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        max_edges = 1608  # From precomputation
        max_angles = 48824  # From precomputation
        n_species = 10

        _, GNN_energy = neural_networks.dimenetpp_neighborlist_ANI1x(
            displacement, r_cut, n_species, positions_test=None, neighbor_test=None,
            max_edges=max_edges, max_angles=max_angles, kbt_dependent=False, embed_size=64)


        def energy_fn_template(energy_params):
            # Took out species in partial() to force that species is passed, as this varies.
            gnn_energy = partial(GNN_energy, energy_params)

            def energy(R, neighbor, species, **dynamic_kwargs):
                return gnn_energy(positions=R, neighbor=neighbor, species=species, **dynamic_kwargs)

            return energy

        with open(path_energy_params, 'rb') as pickle_file:
            loaded_params = pickle.load(pickle_file)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)
        energy_fn_init = energy_fn_template(energy_params=loaded_params)

        # @jax.jit
        # def update_energy_forces(nbrs_passed, pos_passed, species_passed):
        #     nbrs_init = neighbor_fn.update(pos_passed, nbrs_passed)
        #     pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos=pos_passed, neighbor=nbrs_passed,
        #                                                                species=species_passed)
        #     return nbrs_init, pred_energies, neg_forces

        # Load test dataset:
        seed_set = 1
        pad_energies = onp.load('Train_val_test_data/energies_train_' + str(seed_set) + '_size' + str(all_size) + '.npy')
        pad_forces = onp.load('Train_val_test_data/pad_forces_train_' + str(seed_set) + '_size' + str(all_size) + '.npy')
        pad_pos = onp.load('Train_val_test_data/pad_pos_train_seed' + str(seed_set) + '_size' + str(all_size) + '.npy')

        energies_test = pad_energies[:5]

        # Iterate over dataset
        energy_diff_list = []
        force_diff_list = []
        for i in range(len(energies_test)):
            pos = pad_pos[i]
            nbrs_init = neighbor_fn.update(pos, nbrs_init)

            # Predict energy
            pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos, neighbor=nbrs_init,
                                                                       species=pad_species[i])

            num_atoms = onp.count_nonzero(pad_species[i])
            diff_energy_sqr = ((pad_energies[i] / num_atoms) - (pred_energies / num_atoms)) ** 2
            print("Deviation: {} % ".format(((((pad_energies[i] / num_atoms) - (pred_energies / num_atoms))) / (pad_energies[i] / num_atoms))*100))
            print("Predicted energy: ", pred_energies)
            print("True energy: ", pad_energies[i])
            energy_diff = pred_energies - pad_energies[i]
            energy_diff_list.append(energy_diff)

            force_diff = neg_forces + pad_forces[i]
            force_diff_list.append(force_diff)
            print('Done with {} diff'.format(i))
        sq_energies = [i**2 for i in energy_diff_list]
        print("RMSE: [kJ/mol]", ((1 / len(sq_energies)) * sum(sq_energies))**0.5)




