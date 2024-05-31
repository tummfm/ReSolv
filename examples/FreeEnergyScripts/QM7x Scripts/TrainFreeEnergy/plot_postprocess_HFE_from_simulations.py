
import os
import sys
import gc

import pickle
import matplotlib.pyplot as plt
import numpy as onp

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 0


# GPU = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


load_trajs = True
# t_prod = 250
# t_equil = 50
print_every_time = 5
initial_lr = 0.000001
# initial_lr = 0.0000001
lr_decay = 0.1

# date = '120324'
# num_epochs = 1000

date = '080524'
num_epochs = 500
t_prod_U_vac = 250
t_equil_U_vac = 50
data_type = "test"

check_convergence = True
use_subset_array = onp.arange(10, 801, 10)

vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                 'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                 'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                 'Params.pkl'
# save_checkpoint = ("020224_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(initial_lr) + "_lrd" + str(lr_decay))
# save_checkpoint = ("040324_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(initial_lr) + "_lrd" + str(lr_decay))
# trained_params_path = 'checkpoints/' + save_checkpoint + f'_epoch499.pkl'

save_checkpoint = (date + "_t_prod_" + str(t_prod_U_vac) + "ps_t_equil_" +
                   str(t_equil_U_vac) + "ps_iL" + str(initial_lr) + "_lrd" +
                   str(lr_decay) + "_epochs" + str(num_epochs)
                   + "_seed7_train_389" + "mem_0.97")
trained_params_path = 'checkpoints/' + save_checkpoint + f'_epoch499.pkl'

# Load or compute trajs

# save_checkpoint = (date + "_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(initial_lr) + "_lrd"
#                    + str(lr_decay) + "_epochs" + str(num_epochs))
# trained_params_path = 'checkpoints/' + save_checkpoint + f'_epoch999.pkl'

# save_pred_path = 'HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +'.pkl'


if not check_convergence:
    save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
                      'dataset_' + data_type + '.pkl')

    with open(save_pred_path, 'rb') as f:
        results = pickle.load(f)

    print(results)
    mols = results.keys()

    gaff_pred = [results[mol]['GAFF_pred'] for mol in mols]
    exp_HFE = [results[mol]['exp'] for mol in mols]
    sim_pred = [results[mol]['predictions'] for mol in mols]

    # Compute MAE
    sim_error_values = [abs(sim_pred[i] - exp_HFE[i]) for i in range(len(sim_pred))]
    mae_gaff = sum([abs(gaff_pred[i] - exp_HFE[i]) for i in range(len(gaff_pred))]) / len(gaff_pred)
    mae_sim = sum([abs(sim_pred[i] - exp_HFE[i]) for i in range(len(sim_pred))]) / len(sim_pred)
    print(f'MAE GAFF: {mae_gaff}')
    print(f'MAE sim: {mae_sim}')

    # Plot sim_pred and gaff_pred vs exp_HFE
    fig, ax = plt.subplots()
    ax.scatter(exp_HFE, gaff_pred, label='GAFF', color='r')
    ax.scatter(exp_HFE, sim_pred, label='HFE training', color='b')
    ax.plot(exp_HFE, exp_HFE, label='Experiment', color='k')
    plt.legend()
    plt.show()

elif check_convergence:
    all_sim_pred = []
    for use_subset in use_subset_array:
        save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
                              'dataset_' + data_type + '_use_subset_'+str(use_subset)+'.pkl')

        with open(save_pred_path, 'rb') as f:
            results = pickle.load(f)

        print(results)
        mols = results.keys()

        gaff_pred = [results[mol]['GAFF_pred'] for mol in mols]
        exp_HFE = [results[mol]['exp'] for mol in mols]
        sim_pred = [results[mol]['predictions'] for mol in mols]
        all_sim_pred.append(sim_pred)

    all_sim_pred = onp.array(all_sim_pred)
    x_axis = use_subset_array * 5

    value_200ps = []
    value_4ns = []
    for i in range(12):
        value_200ps.append(all_sim_pred[:, i][4])
        value_4ns.append(all_sim_pred[:, i][-1])
        plt.plot(x_axis, all_sim_pred[:, i])
        # plt.axvline(x=200, color='c', linestyle='--')
        # plt.show()
    # print("Debug")
    plt.axvline(x=200, color='grey', linestyle='--')
    plt.ylabel("$\Delta A_{sim}$ [kcal/mol]")
    plt.xlabel("simulation time [ps]")
    # plt.show()
    plt.savefig("plot_results/convergence_HFE_simulatiotime.pdf", bbox_inches='tight', format='pdf')

