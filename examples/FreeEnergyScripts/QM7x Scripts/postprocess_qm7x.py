import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import pickle
from jax_md import space, energy
from jax import numpy as jnp, lax, nn, value_and_grad
import matplotlib.pyplot as plt


from chemtrain import neural_networks
from chemtrain.jax_md_mod import custom_space


def plot_true_vs_predicted_energies(true_array, pred_array, save_fig_str=None):
    """Function takes in true forces as true_array, predicted forces as pred_array,
    plots a line true on true forces and the predicted forces as circles."""
    fig, ax = plt.subplots()
    true_array_sorted = sorted(true_array)
    ax.plot(true_array_sorted, true_array_sorted, color='blue', linestyle='-')
    ax.scatter(true_array, pred_array, color='red', marker='o')
    ax.set_xlabel('True energies')  # [eV]')
    ax.set_ylabel('Predicted energies')  # [eV]')
    ax.set_title('True vs predicted energies')
    plt.show()
    return


def plot_all_true_vs_predicted_forces(true_total, pred_total, save_fig_str=None):
    """Function takes in true_total(3 lists with x,y,z components of true forces)
    and pred_total(3 lists with x,y,z components fo predicted forces). Plots
    x forces in red, y forces purple, and z forces in cyan"""
    fig, ax = plt.subplots()
    all_true_values = []
    for i in true_total:
        for j in i:
            all_true_values.append(j)
    all_true_values.sort()
    ax.plot(all_true_values, all_true_values, color='blue', linestyle='-')
    ax.scatter(true_total[0], pred_total[0], color='red', marker='o', alpha=1, label='x-forces')
    ax.scatter(true_total[1], pred_total[1], color='purple', marker='d', alpha=0.7, label='y-forces')
    ax.scatter(true_total[2], pred_total[2], color='cyan', marker='s', alpha=0.4, label='z-forces')
    ax.legend()
    ax.set_xlabel('True forces')  # [eV/Å]')
    ax.set_ylabel('Predicted forces')  # [eV/Å]')
    ax.set_title('True vs predicted forces')
    plt.show()

if __name__ == '__main__':
    compute_RMSE = True
    scale_amber = False
    remove_100k_largest_amber = True

    if compute_RMSE:

        load_energies = onp.load("QM7x_DB/atom_energies_QM7x.npy")
        all_size = load_energies.shape[0]

        # amber_bool = False
        # date = "210422"
        # id_num = "Nequip_ID_QM7x_4"

        # amber_bool = True
        # date = "280523"
        # id_num = "Nequip_ID_QM7x_Prior_1"

        # amber_bool = True
        # date = "150623"
        # id_num = "Nequip_ID_QM7x_Prior_Precomp_4"

        amber_bool = True
        date = "230623"
        id_num = "Nequip_ID_QM7x_Prior_Precomp_12"
        # id_num = "Nequip_ID_QM7x_Prior_Precomp_23"

        num_epochs = 20
        shift_b = "False"
        scale_b = "False"
        mlp = "4"
        train_on = "EnergiesAndForces"

        save_fig_energy = "Postprocessing_plots/" + date + "_" + id_num + "_energy.png"
        save_fig_force = "Postprocessing_plots/" + date + "_" + id_num + "_force.png"
        # save_name = "210422_QM7x_Nequip_" + id_num + "_Trainer_ANI1x_Energies" + str(all_size) + "samples_" + str(
        #     num_epochs) + "epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp" + mlp + "_Shift" + shift_b + "_Scale" + scale_b + "_" + train_on

        save_name = date + "_QM7x_Nequip_" + id_num + "_Trainer_ANI1x_Energies" + str(all_size) + "samples_" + str(
            num_epochs) + "epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp" + mlp + "_Shift" + shift_b + "_Scale" + \
                    scale_b + "_" + train_on

        # best_params_path = '/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/savedTrainers/' + str(
        #     save_name) + '_bestParams.pkl'

        save_path = date+"_QM7x_Nequip_"+id_num+"_Trainer_ANI1x_Energies"+str(all_size)+"samples_" + str(
        num_epochs) + "epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp"+mlp+"_Shift"+shift_b+"_Scale"+scale_b+"_"+train_on
        best_params_path = '/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/savedTrainers/' + str(save_path)+'_epoch_11_bestParams.pkl'


        nequip_units = True

        energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")[:all_size]
        pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[:all_size]
        pad_forces = onp.load("QM7x_DB/shuffled_atom_forces_QM7x.npy")[:all_size]
        pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[:all_size]

        # # Load amber energies
        # amber_energy = onp.load("QM7x_DB/shuffled_amber_energies_QM7x.npy")[:all_size]

        amber_energy = onp.load("QM7x_DB/shuffled_amber_energies_QM7x_0_to_4195237.npy")[:all_size]
        amber_force = onp.load("QM7x_DB/shuffled_amber_forces_QM7x_0_to_4195237.npy")[:all_size]

        if scale_amber:
            # Scaling amber energies and forces
            amber_energy /= 1000
            amber_force /= 1000

        if remove_100k_largest_amber:
            # Remove 10k largest amber energies
            remove_max_amber_energy_indices = onp.argsort(amber_energy)[-100000:]
            max_amber_forces = onp.array([onp.max(onp.abs(force)) for force in amber_force])
            remove_max_amber_forces_indices = onp.argsort(max_amber_forces)[-100000:]

            # Check how many indices are the same in both lists
            print("Number of indices that are the same in both lists: ",
                  len(onp.intersect1d(remove_max_amber_energy_indices, remove_max_amber_forces_indices)))

            # Combine the two lists and remove duplicates
            remove_max_amber_indices = onp.unique(
                onp.concatenate((remove_max_amber_energy_indices, remove_max_amber_forces_indices), axis=0))
            print("Length of remove_max_amber_indices: ", len(remove_max_amber_indices))
            print("Debug")

            # if assert not true print error
            if not amber_energy.shape[0] == energies.shape[0]:
                print("ERROR: amber_energy.shape[0] != energies.shape[0]")
                sys.exit()

            # Remove the indices from the data
            energies = onp.delete(energies, remove_max_amber_indices, axis=0)
            pad_pos = onp.delete(pad_pos, remove_max_amber_indices, axis=0)
            pad_forces = onp.delete(pad_forces, remove_max_amber_indices, axis=0)
            pad_species = onp.delete(pad_species, remove_max_amber_indices, axis=0)
            amber_energy = onp.delete(amber_energy, remove_max_amber_indices, axis=0)
            amber_force = onp.delete(amber_force, remove_max_amber_indices, axis=0)


        # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
        pad_species = onp.array(pad_species, dtype='int32')


        if nequip_units:
            for i, pos in enumerate(pad_pos):
                pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])

            for i, pos in enumerate(pad_pos):
                x_spacing = 0
                for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                    x_spacing += 15.0
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
        config = neural_networks.initialize_nequip_cfg_MaxSetup(n_species, r_cut)
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

        def energy_fn_template_amber(energy_params):
            def energy_fn(pos, neighbor, amber_part, **dynamic_kwargs):
                _species = dynamic_kwargs.pop('species', None)
                if _species is None:
                    raise ValueError('Species needs to be passed to energy_fn')
                atoms_comp = nn.one_hot(_species, n_species)
                gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                           atoms=atoms_comp, **dynamic_kwargs) + amber_part
                # prior_energy = prior_potential(prior_fns, pos, neighbor,
                #                                **dynamic_kwargs)
                return gnn_energy

            return energy_fn

        with open(best_params_path, 'rb') as pickle_file:
            loaded_params = pickle.load(pickle_file)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        if amber_bool:
            energy_fn_init = energy_fn_template_amber(energy_params=loaded_params)
        else:
            energy_fn_init = energy_fn_template(energy_params=loaded_params)

        # @jax.jit
        # def update_energy_forces(nbrs_passed, pos_passed, species_passed):
        #     nbrs_init = neighbor_fn.update(pos_passed, nbrs_passed)
        #     pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos=pos_passed, neighbor=nbrs_passed,
        #                                                                species=species_passed)
        #     return nbrs_init, pred_energies, neg_forces

        # Get test set
        # len_energy = len(energies)
        len_energy = 20
        start_test = len_energy - 20
        energies_test = energies[start_test:len_energy]

        # Iterate over dataset
        energy_diff_list = []
        force_diff_list = []
        all_pred_energies = []
        all_true_energies = []
        all_pred_forces = []
        all_true_forces = []
        energy_diff_per_atom_list = []

        for i in range(start_test, len(energies)):
            pos = pad_pos[i]
            amber_energy_part = amber_energy[i]
            amber_force_part = amber_force[i]
            nbrs_init = neighbor_fn.update(pos, nbrs_init)

            # Predict energy
            if amber_bool:
                pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos, neighbor=nbrs_init,
                                                                           amber_part=amber_energy_part,
                                                                           species=pad_species[i])
                neg_forces += amber_force_part
            else:
                pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos, neighbor=nbrs_init,
                                                                           species=pad_species[i])

            all_pred_energies.append(pred_energies)
            all_true_energies.append(energies[i])
            all_pred_forces.append(neg_forces)
            all_true_forces.append(pad_forces[i])
            print("Predicted energy: ", pred_energies)
            print("True energy: ", energies[i])
            print("Energy diff: ", pred_energies - energies[i])
            energy_diff = pred_energies - energies[i]
            energy_diff_list.append(energy_diff)
            # print(energy_diff_list)

            force_diff = neg_forces + pad_forces[i]
            force_diff_list.append(force_diff)
            # print('Done with {} diff'.format(i))
            # print(force_diff_list)

            # Compute energy RMSE
            num_atoms = onp.count_nonzero(pad_species[i])
            energy_diff_per_atom = energy_diff / num_atoms
            energy_diff_per_atom_list.append(energy_diff_per_atom)

        energy_diff_per_atom_list = onp.array(energy_diff_per_atom_list)
        energy_RMSE = onp.linalg.norm(energy_diff_per_atom_list) / onp.sqrt((energy_diff_per_atom_list.shape[0]))
        print("Energy RMSE [eV / atom]: ", energy_RMSE)
        energy_MAE = onp.mean(onp.abs(energy_diff_per_atom_list))
        print("Energy MAE [eV / atom]: ", energy_MAE)

        plot_true_vs_predicted_energies(onp.array(all_true_energies), onp.array(all_pred_energies),
                                        save_fig_str=save_fig_energy)

        # Reshape forces for plotting
        true_force_array = [[], [], []]
        pred_force_array = [[], [], []]
        for i in range(len(all_true_forces)):
            num_atoms = int(onp.count_nonzero(all_true_forces[i]) / 3)
            print("Num atoms: ", num_atoms)
            for j in range(3):
                true_force_array[j].append(onp.array(all_true_forces[i])[:num_atoms, j])
                pred_force_array[j].append(onp.array(all_pred_forces[i])[:num_atoms, j])
        # true_force_array = onp.array(true_force_array)
        # pred_force_array = onp.array(pred_force_array)
        x_list_true = []
        y_list_true = []
        z_list_true = []
        x_list_pred = []
        y_list_pred = []
        z_list_pred = []
        for i in range(len(true_force_array[0])):
            # x_list_true += list(true_force_array[i][0])
            # y_list_true += list(true_force_array[i][1])
            # z_list_true += list(true_force_array[i][2])
            # x_list_pred += list(pred_force_array[i][0])
            # y_list_pred += list(pred_force_array[i][1])
            # z_list_pred += list(pred_force_array[i][2])
            x_list_true += list(true_force_array[0][i])
            y_list_true += list(true_force_array[1][i])
            z_list_true += list(true_force_array[2][i])
            x_list_pred += list(pred_force_array[0][i])
            y_list_pred += list(pred_force_array[1][i])
            z_list_pred += list(pred_force_array[2][i])

        #print("Shape forces: ", true_force_array.shape)
        print("x_list_true: ", len(x_list_true))
        print("y_list_true: ", len(y_list_true))
        print("z_list_true: ", len(z_list_true))
        true_force_array = onp.array([x_list_true, y_list_true, z_list_true])
        pred_force_array = onp.array([x_list_pred, y_list_pred, z_list_pred])
        plot_all_true_vs_predicted_forces(true_total=true_force_array, pred_total=-pred_force_array,
                                          save_fig_str=save_fig_force)

        true_force_array.reshape(-1)
        pred_force_array.reshape(-1)

        force_rmse = onp.linalg.norm((true_force_array - pred_force_array)) / onp.sqrt(true_force_array.shape[0])
        print("Force RMSE [ev / Å]: ", force_rmse)

        force_MAE = onp.mean(onp.abs(true_force_array - pred_force_array))
        print("Force MAE [ev / Å]: ", force_MAE)

