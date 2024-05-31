import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = ""
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import pickle
from jax_md import space, energy
from jax import numpy as jnp, lax, nn, value_and_grad
import matplotlib.pyplot as plt
from jax import jit
import pickle


from chemtrain import neural_networks
from chemtrain.jax_md_mod import custom_space

import anton_scripts.anton_training_utils as HFE_util

from jax import config
config.update("jax_enable_x64", True)


def plot_true_vs_predicted_energies(true_array, pred_array, save_fig_str=None):
    """Function takes in true forces as true_array, predicted forces as pred_array,
    plots a line true on true forces and the predicted forces as circles."""
    fig, ax = plt.subplots()
    true_array_sorted = sorted(true_array)
    ax.plot(true_array_sorted, true_array_sorted, color='k', linestyle='-', label='Experiment')
    ax.scatter(true_array, pred_array, color='blue', marker='o', label='Direct HFE Learning')
    ax.set_xlabel('HFE [kcal/mol]')  # [eV]')
    ax.set_ylabel('HFE [kcal/mol]')  # [eV]')
    plt.legend()

    if save_fig_str is not None:
        plt.savefig(save_fig_str)
    plt.show()
    return

def plot_true_vs_predicted_direct_and_sim_energies(true_energy_direct, pred_energy_direct, exp_HFE, sim_pred,
                                                   save_fig_str=None):
    """Function takes in true forces as true_array, predicted forces as pred_array,
    plots a line true on true forces and the predicted forces as circles."""
    fig, ax = plt.subplots()
    true_array_sorted = sorted(true_energy_direct)
    ax.plot(true_array_sorted, true_array_sorted, color='k', linestyle='-', label='Experiment')
    ax.scatter(true_energy_direct, pred_energy_direct, color='blue', marker='o', label='Direct HFE Learning')
    ax.scatter(exp_HFE, sim_pred, color='red', marker='o', label='MLP HFE Learning')
    ax.set_xlabel('HFE [kcal/mol]')  # [eV]')
    ax.set_ylabel('HFE [kcal/mol]')  # [eV]')
    plt.legend()

    if save_fig_str is not None:
        plt.savefig(save_fig_str, bbox_inches='tight', format='pdf')
        # save as pdf

    plt.show()
    return


if __name__ == '__main__':
    postprocess_hyperparameter_validation_loss = False
    # seed = 20  # or 20 -> also include in name
    postprocess_Nequip_HFE_matching = True
    use_train_data = False
    use_train_and_validation = False
    check_test_errors = False
    plot_direct_HFE_for_paper = False
    if postprocess_Nequip_HFE_matching:
        # date = "090224"
        # date = "290424"
        # date = "120524"
        date = "220524"
        # initial_lr = 1e-2  # float(sys.argv[2])
        # lr_decay = 1e-4  # float(sys.argv[3])
        # initial_lr = 5e-3  # float(sys.argv[2])
        # lr_decay = 5e-4  # float(sys.argv[3])
        # batch_size = 2  # int(sys.argv[4])
        # num_epochs = 200  # int(sys.argv[5])
        # seed = 10

        initial_lr = 1e-3  # float(sys.argv[2])
        lr_decay = 1e-3  # float(sys.argv[3])
        batch_size = 2  # int(sys.argv[4])
        num_epochs = 400  # int(sys.argv[5])
        seed = 10

        best_params = False

        # id_num = "Nequip_HFE_matching_date"+str(date)+"_iL" + str(initial_lr) + "_lrd" + str(lr_decay) + "_epochs"\
        #          + str(num_epochs)
        # id_num = ("Nequip_HFE_matching_date" + str(date) + "_iL" + str(initial_lr) + "_lrd" + str(
        #     lr_decay) + "_epochs" + str(num_epochs) + '_batch_size' + str(batch_size))
        id_num = ("Nequip_HFE_matching_date" + str(date) + "_iL" + str(initial_lr) + "_lrd" + str(
            lr_decay) + "_epochs" + str(num_epochs) + '_batch_size' + str(batch_size) + "seed_" + str(seed)
                  + "_fold_number_0")

        load_trainer = "savedTrainers/" + date + "_" + id_num + ".pkl"

        # Load data
        # data_indices = HFE_util.get_404_list()
        # data_indices = HFE_util.get_train_dataset()
        data_indices = HFE_util.get_389_train_dataset()
        # exclusion_list = None
        # exclusion_list = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]

        dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479,
                                        549]
        dataset_for_failed_U_wat = [30, 190, 280, 294, 420]
        exclusion_list = dataset_for_failed_molecules + dataset_for_failed_U_wat
        no_confs = 1

        if use_train_data or use_train_and_validation:
            data, Fe_values, smiles, _, exp_uc = HFE_util.generate_data(data_indices, no_confs, exclusion_list, "train")
        else:
            data, Fe_values, smiles, _, exp_uc = HFE_util.generate_data(data_indices, no_confs, exclusion_list, "test")
        pos = [data[i][0] for i in range(len(data))]
        species = [data[i][2] for i in range(len(data))]

        # Pad pos data with zeros
        max_len = max([len(pos[i]) for i in range(len(pos))])
        for i in range(len(pos)):
            pos[i] = jnp.pad(pos[i], ((0, max_len - len(pos[i])), (0, 0)))
            species[i] = jnp.pad(species[i], (0, max_len - len(species[i])))
        pad_pos = onp.array(pos)
        pad_species = onp.array(species)
        energies = onp.array(Fe_values)
        exp_uc = onp.array(exp_uc)

        # seed = 20
        # indices = onp.arange(len(pad_pos))
        # onp.random.shuffle(indices)
        # pad_pos = pad_pos[indices]
        # pad_species = pad_species[indices]
        # energies = energies[indices]
        # exp_uc = exp_uc[indices]

        max_len_mol_idx = onp.argmax([onp.count_nonzero(pad_pos[i, :, 2]) for i in range(len(pos))])

        if use_train_data:
            save_fig_energy = "results_direct_HFE/train_" + date + "_" + id_num + "_energy.png"
        elif use_train_and_validation:
            save_fig_energy = "results_direct_HFE/train_and_validation_" + date + "_" + id_num + "_energy.png"
        else:
            save_fig_energy = "results_direct_HFE/test_" + date + "_" + id_num + "_energy.png"

        # Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops
        pad_species = onp.array(pad_species, dtype='int32')

        if use_train_data:
            train_ratio = 0.9
            train_len = int(train_ratio * len(pad_pos))
            energies = energies[:train_len]
            pad_pos = pad_pos[:train_len]
            pad_species = pad_species[:train_len]
        amber_energy = onp.zeros_like(energies)

        # Pad positions
        # for i, pos in enumerate(pad_pos):
        #     pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])

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
        init_pos = pad_pos[max_len_mol_idx]   # Important for initialize neighborlist
        init_species = pad_species[max_len_mol_idx]

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
                return gnn_energy

            return energy_fn


        with open(load_trainer, 'rb') as pickle_file:
            loaded_trainer = pickle.load(pickle_file)
            if best_params:
                loaded_params = loaded_trainer.best_params
            else:
                print("Use last params not best params")
                loaded_params = loaded_trainer.params

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)
        energy_fn_init = energy_fn_template(energy_params=loaded_params)

        start_test = 0

        # Iterate over dataset
        energy_diff_list = []
        all_pred_energies = []
        all_true_energies = []
        energy_diff_per_atom_list = []
        num_atom_list = []
        exp_uc_list = []
        smiles_list = []


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
            pred_energies, _, nbrs_init = compute_energy_and_forces_update_nbrs(pos, nbrs_init, amber_energy_part,
                                                                                pad_species[i])


            all_pred_energies.append(pred_energies)
            all_true_energies.append(energies[i])
            exp_uc_list.append(exp_uc[i])
            smiles_list.append(smiles[i])



            print("Predicted energy: ", pred_energies)
            print("True energy: ", energies[i])
            print("Energy diff: ", pred_energies - energies[i])
            print("Num atoms: ", jnp.count_nonzero(pad_species[i]))
            energy_diff = pred_energies - energies[i]
            energy_diff_list.append(energy_diff)
            num_atom_list.append(jnp.count_nonzero(pad_species[i]))

            # # Compute energy RMSE
            # num_atoms = onp.count_nonzero(pad_species[i])
            # energy_diff_per_atom = energy_diff / num_atoms
            # energy_diff_list.append(energy_diff_per_atom)
        energy_diff_list = onp.array(energy_diff_list)
        energy_RMSE = onp.linalg.norm(energy_diff_list) / onp.sqrt((energy_diff_list.shape[0]))
        print("Energy RMSE [kcal / mol]: ", energy_RMSE)
        energy_MAE = onp.mean(onp.abs(energy_diff_list))
        print("Energy MAE [kcal / mol]: ", energy_MAE)

        all_information = [onp.array(all_true_energies), onp.array(all_pred_energies), onp.array(exp_uc_list), smiles_list]
        if use_train_data:
            onp.save("results_direct_HFE/train_energy_mae_" + id_num + ".npy", energy_MAE)
            onp.save("results_direct_HFE/train_true_vs_pred_energy_uc_and_" + id_num + ".npy",
                     onp.array([onp.array(all_true_energies), onp.array(all_pred_energies), onp.array(exp_uc_list)]))
            with open("results_direct_HFE/train_all_information_" + id_num + ".npy", "wb") as file:
                pickle.dump(all_information, file)
        elif use_train_and_validation:
            onp.save("results_direct_HFE/train_and_validation_energy_mae_" + id_num + ".npy", energy_MAE)
            onp.save("results_direct_HFE/train_and_validation_true_vs_pred_energy_uc_and_" + id_num + ".npy",
                     onp.array([onp.array(all_true_energies), onp.array(all_pred_energies), onp.array(exp_uc_list)]))
            with open("results_direct_HFE/train_and_validation_all_information_" + id_num + ".npy", "wb") as file:
                pickle.dump(all_information, file)
        else:
            onp.save("results_direct_HFE/test_energy_mae_" + id_num + ".npy", energy_MAE)
            onp.save("results_direct_HFE/test_true_vs_pred_energy_uc_and_" + id_num + ".npy",
                     onp.array([onp.array(all_true_energies), onp.array(all_pred_energies), onp.array(exp_uc_list)]))
            with open("results_direct_HFE/test_all_information_" + id_num + ".npy", "wb") as file:
                pickle.dump(all_information, file)

        plot_true_vs_predicted_energies(onp.array(all_true_energies), onp.array(all_pred_energies),
                                        save_fig_str=save_fig_energy)

    if postprocess_hyperparameter_validation_loss:
        # date = "090224"
        # date = "290424"
        date = "220524"
        num_epochs = 500  # Either 50, 100
        batch_size = 5  # Either 2, 5
        fold_number_list = [1, 2, 3, 4, 5]
        init_lr_list = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        lr_decay_list = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        min_val_loss_list = []
        seed = 10

        for count in range(len(init_lr_list)):
            min_val_loss_temp = 0
            for fold_number in fold_number_list:
                initial_lr = init_lr_list[count]
                lr_decay = lr_decay_list[count]
                id_num = ("Nequip_HFE_matching_date" + str(date) + "_iL" + str(initial_lr) + "_lrd" + str(
                    lr_decay) + "_epochs" + str(num_epochs) + '_batch_size' + str(batch_size) + "seed_" +
                          str(seed) + "_fold_number_" + str(fold_number))
                save_path = "savedTrainers/" + date + "_" + id_num + ".pkl"
                with open(save_path, 'rb') as pickle_file:
                    loaded_trainer = pickle.load(pickle_file)
                # Evalute validation loss
                # min_val_loss = min(loaded_trainer.val_losses)
                min_val_loss_temp += loaded_trainer.val_losses[-1]

            min_val_loss_list.append(min_val_loss_temp / len(fold_number_list))
        #
        min_val_loss_list = onp.array(min_val_loss_list)
        print("Min val loss list: ", min_val_loss_list)
        min_arg = onp.argmin(min_val_loss_list)
        min_init_lr = init_lr_list[min_arg]
        min_lr_decay = lr_decay_list[min_arg]
        print("Minium at init lr: ", min_init_lr, " and lr decay: ", min_lr_decay)

    if check_test_errors:
        # date = "090224"
        date = "050324"
        initial_lr_list = [5e-3]  # [5e-3, 5e-3, 5e-4, 5e-2, 5e-3, 1e-2, 1e-3, 5e-3]
        lr_decay_list = [5e-4]  # [5e-4, 5e-4, 5e-3, 5e-5, 5e-4, 1e-4, 1e-3, 5e-4]
        batch_size_list = [5] # [2, 5, 2, 5, 2, 5, 2, 5]
        num_epochs_list = [400]  #[50, 50, 100, 100, 200, 200, 300, 300]



        for i in range(len(initial_lr_list)):
            initial_lr = initial_lr_list[i]
            lr_decay = lr_decay_list[i]
            batch_size = batch_size_list[i]
            num_epochs = num_epochs_list[i]
            id_num = ("Nequip_HFE_matching_date" + str(date) + "_iL" + str(initial_lr) + "_lrd" + str(
                lr_decay) + "_epochs" + str(num_epochs) + '_batch_size' + str(batch_size))

            if use_train_data:
                energy_mae = onp.load("results_direct_HFE/train_energy_mae_" + id_num + ".npy")
            else:
                energy_mae = onp.load("results_direct_HFE/test_energy_mae_" + id_num + ".npy")

            print("Energy MAE: ", energy_mae, " for iL = ", initial_lr, " and lr_decay = ", lr_decay, " epochs = ",
                  num_epochs, " batch_size = ", batch_size)

    if plot_direct_HFE_for_paper:
        # date = "090224"
        date = "050324"
        initial_lr = 5e-3
        lr_decay = 5e-4
        batch_size = 2
        num_epochs = 200

        id_num = ("Nequip_HFE_matching_date" + str(date) + "_iL" + str(initial_lr) + "_lrd" + str(
            lr_decay) + "_epochs" + str(num_epochs) + '_batch_size' + str(batch_size))
        # id_num = "Nequip_HFE_matching_date090224_iL0.005_lrd0.0005_epochs200_batch_size2"
        # true_energy_direct, pred_energy_direct = onp.load("results_direct_HFE/test_true_vs_pred_energy_and_uc_" + id_num + ".npy")
        true_energy_direct, pred_energy_direct, exp_uncertrainty = onp.load(
            "results_direct_HFE/test_true_vs_pred_energy_and_uc_" + id_num + ".npy")
        print("Exp uncertainty: ", exp_uncertrainty)

        # plot_true_vs_predicted_energies(true_energy_direct, pred_energy_direct, save_fig_str=None)

        load_trajs = True
        t_prod = 25
        t_equil = 5
        print_every_time = 0.5
        initial_lr = 0.000001
        lr_decay = 0.1
        vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                         'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                         'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                         'Params.pkl'
        save_checkpoint = ("020224_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(
            initial_lr) + "_lrd" + str(lr_decay))
        trained_params_path = 'TrainFreeEnergy/checkpoints/' + save_checkpoint + f'_epoch499.pkl'
        # Load or compute trajs
        save_pred_path = ('TrainFreeEnergy/HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_'
                          + str(load_trajs) + '.pkl')
        with open(save_pred_path, 'rb') as f:
            results = pickle.load(f)

        mols = results.keys()
        gaff_pred = [results[mol]['GAFF_pred'] for mol in mols]
        exp_HFE = onp.array([results[mol]['exp'] for mol in mols])
        sim_pred = onp.array([results[mol]['predictions'] for mol in mols])

        save_fig_path = "results_direct_HFE/test_true_vs_direct_vs_sim_pred_energy_" + id_num + ".pdf"
        plot_true_vs_predicted_direct_and_sim_energies(true_energy_direct, pred_energy_direct,
                                                       exp_HFE, sim_pred, save_fig_str=None)

