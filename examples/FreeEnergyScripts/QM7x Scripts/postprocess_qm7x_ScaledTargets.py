import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import pickle
from jax_md import space, energy
from jax import numpy as jnp, lax, nn, value_and_grad
import matplotlib.pyplot as plt
from jax import jit


from chemtrain import neural_networks
from chemtrain.jax_md_mod import custom_space

from jax import config
config.update("jax_enable_x64", True)


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


def unscale_forces_and_energies(energy_passed, force_passed, amber_energy_passed, amber_force_passed,
                                mean_energy, std_energy):
    energy_passed *= std_energy
    energy_passed += mean_energy
    energy_passed += amber_energy_passed

    force_passed *= std_energy
    force_passed += amber_force_passed

    return energy_passed, force_passed


if __name__ == '__main__':
    compute_RMSE = True
    if compute_RMSE:

        use_only_equilibrium = False
        subset_equilibrium = False
        shift_and_scale_per_atom = False
        use_amber = False
        scale_target = True
        # date = "260823"
        # date = "070923"
        # date = "021023"
        # date = "091023"
        date = "261023"
        # date = "281123"

        # load_energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")
        if use_only_equilibrium:
            load_energies = onp.load("QM7x_DB/equilibrium_shuffled_atom_energies_QM7x.npy")
        else:
            load_energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")
        all_size = load_energies.shape[0]

        # train_size = int(all_size * 0.7)
        amber_bool = False

        test_size = 10100
        val_ratio = 0.1
        train_ratio = 1 - 0.1 - 0.0024073
        # train_ratio = 1 - 0.1 - 0.0024073
        # val_ratio = 0.1
        # train_size = int(train_ratio * all_size)
        # val_size = int(val_ratio * all_size)

        # id_num = "Nequip_QM7x_All_8epochs_iL5emin3"
        # num_epochs = 8

        # id_num = "Nequip_QM7x_All_WithAmber_remove100klargest_4pochs_iL5emin3_lrdecay5emin3"
        # num_epochs = 4

        # id_num = "Nequip_QM7x_All_WithAmber_remove100klargest_2epochs_iL1emin2_lrdecay1emin3"
        # num_epochs = 2

        # id_num = "Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet"
        # num_epochs = 8

        lr_decay = 1e-3
        initial_lr = 5e-3
        num_epochs = 8
        id_num = "Nequip_QM7x_All_WithAmber_" + str(num_epochs) + "epochs_iL" + str(initial_lr) + "_lrdecay" + str(
            lr_decay) + "_scaledTargets_LargeTrainingSet_Cutoff4A"

        # std_energy = avg_force_rms
        keep_100k_smallest_amber = False
        remove_100k_largest_amber = False

        shift_b = "False"
        scale_b = "False"
        mlp = "4"
        train_on = "EnergiesAndForces"

        save_fig_energy = "Postprocessing_plots/" + date + "_" + id_num + "_energy.png"
        save_fig_force = "Postprocessing_plots/" + date + "_" + id_num + "_force.png"

        # save_name = date + "_QM7x_Nequip_" + id_num + "_Trainer_ANI1x_Energies" + str(all_size) + "samples_" + str(
        #     num_epochs) + "epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp" + mlp + "_Shift" + shift_b + "_Scale" + \
        #             scale_b + "_" + train_on + "_epoch_3"

        save_name = date + "_QM7x_Nequip_" + id_num + "_Trainer_ANI1x_Energies" + str(
            num_epochs) + "epochs_mlp" + mlp + "_Shift" + shift_b + "_Scale" + scale_b + "_" + train_on + "_epoch_8"

        best_params_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/' + str(
            save_name) + '_Params.pkl'

        best_params_path = ('/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_'
                            'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_'
                            'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_'
                            'Params.pkl')

        if use_only_equilibrium:
            energies = onp.load("QM7x_DB/equilibrium_shuffled_atom_energies_QM7x.npy")[:all_size]
            pad_pos = onp.load("QM7x_DB/equilibrium_shuffled_atom_positions_QM7x.npy")[:all_size]
            pad_forces = onp.load("QM7x_DB/equilibrium_shuffled_atom_forces_QM7x.npy")[:all_size]
            pad_species = onp.load("QM7x_DB/equilibrium_shuffled_atom_numbers_QM7x.npy")[:all_size]

        else:
            energies = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy")[:all_size]
            pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[:all_size]
            pad_forces = onp.load("QM7x_DB/shuffled_atom_forces_QM7x.npy")[:all_size]
            pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[:all_size]

        if use_amber:
            amber_energy = onp.load("QM7x_DB/shuffled_amber_energies_QM7x_0_to_4195237.npy")[:all_size]
            amber_force = onp.load("QM7x_DB/shuffled_amber_forces_QM7x_0_to_4195237.npy")[:all_size]
        else:
            amber_energy = onp.zeros_like(energies)
            amber_force = onp.zeros_like(pad_forces)

        if subset_equilibrium:
            energies = energies[:100]
            pad_pos = pad_pos[:100]
            pad_forces = pad_forces[:100]
            pad_species = pad_species[:100]
            amber_energy = amber_energy[:100]
            amber_force = amber_force[:100]
            train_size = int(energies.shape[0] * 0.7)
            energies = energies[:70]
            pad_pos = pad_pos[:70]
            pad_forces = pad_forces[:70]
            pad_species = pad_species[:70]
            amber_energy = amber_energy[:70]
            amber_force = amber_force[:70]

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

        all_size = energies.shape[0]
        test_ratio = (test_size / all_size) - 1e-7
        train_ratio = 1 - test_ratio - val_ratio
        train_size = int(all_size * train_ratio)
        print("Test size: ", test_size)
        # if test_size != (all_size - int(train_ratio * all_size) - int(val_ratio * all_size)):
        #     print("ERROR: Test size does not match intended size")
        #     sys.exit()

        if keep_100k_smallest_amber:
            idx = onp.argsort(amber_energy)[:100000]
            energies = energies[idx]
            pad_pos = pad_pos[idx]
            pad_forces = pad_forces[idx]
            pad_species = pad_species[idx]
            amber_energy = amber_energy[idx]
            amber_force = amber_force[idx]


        if shift_and_scale_per_atom and not use_amber:
            num_atoms = onp.count_nonzero(pad_species[:train_size])
            sum_energy = onp.sum(energies[:train_size])

            # Compute scale and shift
            mean_per_atom_energy = sum_energy / num_atoms
            avg_std_force = onp.mean([onp.std(force_single) for force_single in pad_forces[:train_size]])
            shift_U = mean_per_atom_energy
            scale_U_F = avg_std_force
        elif shift_and_scale_per_atom and use_amber:
            energies = energies - amber_energy
            pad_forces = pad_forces - amber_force

            num_atoms = onp.count_nonzero(pad_species[:train_size])
            sum_energy = onp.sum(energies[:train_size])

            # Compute scale and shift
            mean_per_atom_energy = sum_energy / num_atoms
            avg_std_force = onp.mean([onp.std(force_single) for force_single in pad_forces[:train_size]])
            shift_U = mean_per_atom_energy
            scale_U_F = avg_std_force

            energies = energies + amber_energy
            pad_forces = pad_forces + amber_force
        elif scale_target:
            # Modify the target energies + forces by subtracting the amber contribution
            energies = energies - amber_energy
            pad_forces = pad_forces - amber_force

            # Do scaling of energies and forces
            # Compute the mean potential energy over the training dataset
            mean_energy = onp.mean(energies[:train_size])

            avg_force_rms = onp.mean(
                [onp.sqrt(onp.mean(force_single ** 2)) for force_single in pad_forces[:train_size]])

            # Set amber energies & forces to 0 for case with amber in target
            amber_energy = onp.zeros_like(amber_energy)
            amber_force = onp.zeros_like(amber_force)
        else:
            print("Missing code")
            sys.exit(0)


        # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
        pad_species = onp.array(pad_species, dtype='int32')

        # Use subset
        energies = energies[-test_size:]
        pad_pos = pad_pos[-test_size:]
        pad_forces = pad_forces[-test_size:]
        pad_species = pad_species[-test_size:]
        amber_energy = amber_energy[-test_size:]
        amber_force = amber_force[-test_size:]


        # Pad positions
        for i, pos in enumerate(pad_pos):
            pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])

        for i, pos in enumerate(pad_pos):
            x_spacing = 0
            for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
                x_spacing += 15.0
                pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing

            box = jnp.eye(3) * 1000

        scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
        pad_pos = lax.map(scale_fn, pad_pos)
        # Turn all arrays to jax.numpy format
        pad_species = jnp.array(pad_species)
        pad_pos = jnp.array(pad_pos)
        init_pos = pad_pos[0]   # Important for initialize neighborlist
        init_species = pad_species[0]

        # Load energy_fn to make predictions
        n_species = 100
        r_cut = 4.0

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
                return gnn_energy

            return energy_fn

        with open(best_params_path, 'rb') as pickle_file:
            loaded_params = pickle.load(pickle_file)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        if amber_bool:
            energy_fn_init = energy_fn_template_amber(energy_params=loaded_params)
        else:
            energy_fn_init = energy_fn_template(energy_params=loaded_params)

        start_test = 0
        energies_test = energies

        # Iterate over dataset
        energy_diff_list = []
        force_diff_list = []
        all_pred_energies = []
        all_true_energies = []
        all_pred_forces = []
        all_true_forces = []
        energy_diff_per_atom_list = []
        num_atom_list = []

        @jit
        def compute_energy_and_forces_update_nbrs(pos, nbrs, amber_energy, species):
            nbrs = neighbor_fn.update(pos, nbrs)
            pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos, neighbor=nbrs,
                                                                       amber_part=amber_energy,
                                                                       species=species)
            return pred_energies, neg_forces, nbrs

        for i in range(start_test, len(energies)):
            pos = pad_pos[i]
            amber_energy_part = onp.zeros_like(amber_energy[i])

            # Predict energy and forces
            # Jitted version
            pred_energies, neg_forces, nbrs_init = compute_energy_and_forces_update_nbrs(pos, nbrs_init,
                                                                                         amber_energy_part,
                                                                                         pad_species[i])
            # unjitted version
            # nbrs_init = neighbor_fn.update(pos, nbrs_init)
            # pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos, neighbor=nbrs_init,
            #                                                            amber_part=amber_energy_part,
            #                                                            species=pad_species[i])

            if shift_and_scale_per_atom and not use_amber:
                num_atoms = jnp.count_nonzero(pad_species[i])
                U_scaled_shifted = pred_energies * scale_U_F + shift_U * num_atoms
                F_scaled = -neg_forces * scale_U_F
                pred_energy = U_scaled_shifted
                pred_force = F_scaled

            elif shift_and_scale_per_atom and use_amber:
                num_atoms = jnp.count_nonzero(pad_species[i])
                U_scaled_shifted = pred_energies * scale_U_F + shift_U * num_atoms
                F_scaled = -neg_forces * scale_U_F
                pred_energy = U_scaled_shifted + amber_energy[i]
                pred_force = F_scaled + amber_force[i]

            elif scale_target:
                pred_force = -neg_forces * avg_force_rms
                pred_energy = pred_energies * avg_force_rms + mean_energy

            else:
                pred_energy, pred_force = unscale_forces_and_energies(pred_energies, -neg_forces, amber_energy[i],
                                                                      amber_force[i], mean_energy=mean_energy,
                                                                      std_energy=std_energy)


            all_pred_energies.append(pred_energy)
            all_true_energies.append(energies[i])
            all_pred_forces.append(pred_force)
            all_true_forces.append(pad_forces[i])
            print("Predicted energy: ", pred_energy)
            print("True energy: ", energies[i])
            print("Energy diff: ", pred_energy - energies[i])
            print("Num atoms: ", jnp.count_nonzero(pad_species[i]))
            energy_diff = pred_energy - energies[i]
            energy_diff_list.append(energy_diff)

            force_diff = pred_force - pad_forces[i]
            force_diff_list.append(force_diff)
            num_atom_list.append(jnp.count_nonzero(pad_species[i]))


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
            #print("Num atoms: ", num_atoms)
            for j in range(3):
                true_force_array[j].append(onp.array(all_true_forces[i])[:num_atoms, j])
                pred_force_array[j].append(onp.array(all_pred_forces[i])[:num_atoms, j])
        x_list_true = []
        y_list_true = []
        z_list_true = []
        x_list_pred = []
        y_list_pred = []
        z_list_pred = []
        for i in range(len(true_force_array[0])):
            x_list_true += list(true_force_array[0][i])
            y_list_true += list(true_force_array[1][i])
            z_list_true += list(true_force_array[2][i])
            x_list_pred += list(pred_force_array[0][i])
            y_list_pred += list(pred_force_array[1][i])
            z_list_pred += list(pred_force_array[2][i])

        print("x_list_true: ", len(x_list_true))
        print("y_list_true: ", len(y_list_true))
        print("z_list_true: ", len(z_list_true))
        true_force_array = onp.array([x_list_true, y_list_true, z_list_true])
        pred_force_array = onp.array([x_list_pred, y_list_pred, z_list_pred])
        plot_all_true_vs_predicted_forces(true_total=true_force_array, pred_total=pred_force_array,
                                          save_fig_str=save_fig_force)

        true_force_array.reshape(-1)
        pred_force_array.reshape(-1)

        force_rmse = onp.linalg.norm((true_force_array - pred_force_array)) / onp.sqrt(true_force_array.shape[0])
        print("Force RMSE [ev / Å]: ", force_rmse)

        force_MAE = onp.mean(onp.abs(true_force_array - pred_force_array))
        print("Force MAE [ev / Å]: ", force_MAE)

        # Reshape onp.abs(true_force_array - pred_force_array) to 1D array
        diff_arr = onp.reshape(true_force_array - pred_force_array, -1)
        diff_arr = onp.abs(diff_arr)
        diff_arr = onp.sort(diff_arr)[:-6]


        force_MAE = onp.mean(diff_arr)
        print("Force MAE [ev / Å]: ", force_MAE)

