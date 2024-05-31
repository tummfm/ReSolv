import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import pickle
from jax_md import space, energy
from jax import numpy as jnp, lax, nn, value_and_grad
import matplotlib.pyplot as plt


from chemtrain import neural_networks
from chemtrain.jax_md_mod import custom_space

if __name__ == '__main__':
    compute_RMSE = True
    if compute_RMSE:

        # path_energy_params = "savedTrainers/310322_Nequip_EnergyParams_ANI1x_EnergiesAndForces1kSubsetOfAll_200epochs_gu2emin7_gf1emin3_iL5emin3_LRdecay1emin3_mlp4.pkl"
        path_energy_params = "savedTrainers/060422_Nequip_New_ID24_Trainer_ANI1x_Energies100287samples_10epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_bestParams.pkl"
        load_dataset_name = 'ANI1xDB/'
        all_size = 100287
        nequip_units = True

        pad_pos = onp.load(load_dataset_name + 'pad_pos.npy')[:all_size]
        pad_species = onp.load(load_dataset_name + 'mask_species.npy')[:all_size]

        # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
        pad_species = onp.array(pad_species, dtype='int32')


        if nequip_units:
            for i, pos in enumerate(pad_pos):
                pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])

            for i, pos in enumerate(pad_pos):
                x_spacing = 0
                for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                    x_spacing += 6.0
                    pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing

            box = jnp.eye(3) * 1000
        else:
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
        n_species = 100
        if nequip_units:
            r_cut = 5.0
        else:
            r_cut = 0.5
        dr_thresh = 0.05
        neighbor_capacity_multiple = 2.7  # Hard coded for ANI1-x dataset.
        displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)

        # TODO : ATTENTION Be aware that config must be the same as with training!
        config = neural_networks.initialize_nequip_cfg(n_species, r_cut)
        config.n_neighbors = 20
        atoms = nn.one_hot(init_species, n_species)
        neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
            displacement, box, config, atoms=None, dr_threshold=dr_thresh,
            capacity_multiplier=neighbor_capacity_multiple,
            fractional_coordinates=True,
            disable_cell_list=True)


        def energy_fn_template(energy_params):
            def energy_fn(pos, neighbor, **dynamic_kwargs):
                _species = dynamic_kwargs.pop('species', None)
                if _species is None:
                    raise ValueError('Species needs to be passed to energy_fn')
                atoms_comp = nn.one_hot(_species, n_species)
                gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                           atoms=atoms_comp, **dynamic_kwargs)
                # prior_energy = prior_potential(prior_fns, pos, neighbor,
                #                                **dynamic_kwargs)
                return gnn_energy

            return energy_fn

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
            print("Predicted energy: ", pred_energies)
            energy_diff = pred_energies - pad_energies[i]
            energy_diff_list.append(energy_diff)
            print(energy_diff_list)

            force_diff = neg_forces + pad_forces[i]
            force_diff_list.append(force_diff)
            print('Done with {} diff'.format(i))
            print(force_diff_list)


