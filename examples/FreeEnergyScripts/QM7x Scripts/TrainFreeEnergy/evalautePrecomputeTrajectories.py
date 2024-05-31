import os
import sys
import typing

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 5

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax import value_and_grad, nn, numpy as jnp, jit
from jax_md import space, energy
import dill
import pickle
import numpy as onp
import matplotlib.pyplot as plt

from chemtrain import neural_networks


def get_404_data():
    _404 = [4, 23, 24, 34, 52, 55, 56, 65, 93, 112, 119, 131, 132, 137, 138, 143, 147, 149, 162, 168, 169, 184, 185,
            189, 193, 202, 231, 242, 245, 270, 272, 274, 276, 278, 279, 295, 322, 326, 327, 329, 331, 335, 392, 393,
            415, 424, 425, 426, 1, 3, 6, 13, 22, 48, 60, 66, 77, 82, 83, 87, 105, 133, 134, 139, 148, 177, 178, 190,
            206, 209, 224, 235, 236, 247, 252, 255, 265, 283, 284, 291, 310, 333, 334, 347, 355, 387, 394, 400, 405,
            407, 409, 418, 21, 103, 156, 176, 188, 260, 275, 319, 342, 352, 359, 364, 19, 113, 165, 210, 246, 244, 9,
            11, 12, 18, 20, 25, 26, 31, 33, 35, 36, 38, 39, 41, 43, 44, 47, 49, 50, 51, 57, 58, 62, 63, 64, 67, 69, 72,
            73, 74, 75, 86, 90, 91, 92, 94, 96, 97, 98, 100, 101, 107, 108, 111, 116, 117, 120, 122, 124, 125, 126, 129,
            136, 142, 146, 150, 152, 157, 158, 160, 161, 163, 166, 174, 175, 182, 186, 187, 191, 195, 198, 205, 211,
            212, 214, 216, 218, 221, 223, 226, 227, 230, 232, 233, 238, 239, 243, 248, 249, 250, 251, 254, 259, 262,
            264, 266, 271, 285, 286, 288, 290, 297, 301, 304, 305, 307, 309, 312, 313, 315, 320, 324, 325, 328, 330,
            336, 337, 338, 341, 345, 346, 351, 354, 358, 360, 362, 363, 368, 369, 370, 374, 377, 378, 380, 381, 383,
            385, 388, 391, 396, 8, 42, 53, 59, 61, 71, 79, 81, 95, 99, 102, 106, 110, 118, 121, 123, 141, 144, 153, 154,
            170, 180, 194, 197, 199, 201, 225, 228, 240, 253, 258, 263, 268, 273, 294, 298, 303, 314, 321, 349, 356,
            375, 382, 398, 401, 406, 427, 453, 0, 17, 30, 32, 37, 54, 80, 84, 128, 155, 213, 217, 219, 256, 267, 282,
            293, 311, 28, 167, 296, 366, 29, 171, 350, 384, 14, 277, 413, 2, 5, 7, 10, 15, 16, 27, 40, 45, 46, 68, 70,
            76, 78, 88, 89, 104, 109, 114, 115, 127, 130, 135, 140, 145, 151, 159, 164, 172, 173, 179, 181, 183, 192,
            196, 200, 203, 204, 207, 208, 215, 220, 222, 229, 234, 237, 241, 261, 269, 280, 281, 287, 289, 292, 300,
            302, 306, 308, 316, 317, 318, 323, 332, 339, 343, 344, 348, 353, 361, 365, 371, 373, 376, 379, 386, 390,
            395, 397]
    return _404

def get_train_dataset():
    train_dataset = [4, 23, 24, 34, 52, 55, 56, 65, 93, 112, 119, 131, 132, 137, 138, 143, 147, 149, 162, 168, 169, 184, 185, 189, 193, 202,
                 231, 242, 245, 270, 272, 274, 276, 278, 279, 295, 322, 326, 327, 329, 331, 335, 392, 393, 415, 424, 425, 426, 1, 3, 6,
                 13, 22, 48, 60, 66, 77, 82, 83, 87, 105, 133, 134, 139, 148, 177, 178, 190, 206, 209, 224, 235, 236, 247, 252, 255, 265,
                 283, 284, 291, 310, 333, 334, 347, 355, 387, 394, 400, 405, 407, 409, 418, 21, 103, 156, 176, 188, 260, 275, 319, 342, 352,
                 359, 364, 19, 113, 165, 210, 246, 244, 9, 11, 12, 18, 20, 25, 26, 31, 33, 35, 36, 38, 39, 41, 43, 44, 47, 49, 50, 51, 57,
                 58, 62, 63, 64, 67, 69, 72, 73, 74, 75, 86, 90, 91, 92, 94, 96, 97, 98, 100, 101, 107, 108, 111, 116, 117, 120, 122, 124,
                 125, 126, 129, 136, 142, 146, 150, 152, 157, 158, 160, 161, 163, 166, 174, 175, 182, 186, 187, 191, 195, 198, 205, 211, 212,
                 214, 216, 218, 221, 223, 226, 227, 230, 232, 233, 238, 239, 243, 248, 249, 250, 251, 254, 259, 262, 264, 266, 271, 285, 286,
                 288, 290, 297, 301, 304, 305, 307, 309, 312, 313, 315, 320, 324, 325, 328, 330, 336, 337, 338, 341, 345, 346, 351, 354, 358,
                 360, 362, 363, 368, 369, 370, 374, 377, 378, 380, 381, 383, 385, 388, 391, 396, 8, 42, 53, 59, 61, 71, 79, 81, 95, 99, 102,
                 106, 110, 118, 121, 123, 141, 144, 153, 154, 170, 180, 194, 197, 199, 201, 225, 228, 240, 253, 258, 263, 268, 273, 294, 298,
                 303, 314, 321, 349, 356, 375, 382, 398, 401, 406, 427, 453, 0, 17, 30, 32, 37, 54, 80, 84, 128, 155, 213, 217, 219, 256, 267,
                 282, 293, 311, 28, 167, 296, 366, 29, 171, 350, 384, 14, 277, 413, 2, 5, 7, 10, 15, 16, 27, 40, 45, 46, 68, 70, 76, 78, 88,
                 89, 104, 109, 114, 115, 127, 130, 135, 140, 145, 151, 159, 164, 172, 173, 179, 181, 183, 192, 196, 200, 203, 204, 207, 208,
                 215, 220, 222, 229, 234, 237, 241, 261, 269, 280, 281, 287, 289, 292, 300, 302, 306, 308, 316, 317, 318, 323, 332, 339, 343,
                 344, 348, 353, 361, 365, 371, 373, 376, 379, 386, 390, 395, 397, 85, 372, 420, 422, 447, 448, 466, 471, 481, 490, 494]
    return train_dataset


def get_train_dataset_389():
    train_dataset_389 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176,
                         181, 187, 188, 205, 206,
                         211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 355, 356, 358, 360, 364,
                         1, 3, 6, 13, 18, 23, 49,
                         63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233, 248, 259,
                         260, 272, 277, 280, 290,
                         310, 311, 318, 337, 362, 363, 366, 377, 385, 22, 112, 173, 195, 210, 285, 302, 347, 372, 382,
                         390, 20, 123, 184, 234,
                         268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50, 51, 52, 57, 59, 60,
                         61, 65, 66, 67, 70, 72,
                         76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107, 109, 110, 116, 117, 120, 121, 127, 128,
                         131, 133, 135, 136, 137,
                         140, 141, 146, 149, 156, 157, 161, 165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203,
                         208, 209, 213, 214, 218,
                         221, 228, 235, 236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273,
                         274, 275, 276, 279, 284,
                         287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343,
                         348, 352, 353, 354, 357,
                         359, 365, 367, 368, 371, 375, 376, 381, 384, 388, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101,
                         103, 108, 111, 115, 119,
                         129, 132, 134, 155, 159, 170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288,
                         295, 300, 321, 325, 330,
                         342, 349, 379, 386, 407, 414, 0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293,
                         294, 29, 186, 323, 30,
                         166, 190, 198, 207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96,
                         113, 118, 124, 125, 138,
                         142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230,
                         231, 232, 239, 244, 246,
                         253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335, 344, 345, 346, 351,
                         361, 369, 373, 374, 378,
                         383, 392, 396, 380, 397, 404, 465, 513]

    return train_dataset_389


def energy_fn_template(energy_params: onp.ndarray) -> typing.Callable:
    """
    Template for energy function.

    Args:
        energy_params: Parameters for energy function.
    """
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


def species_from_mass(mas_array):
    species = []
    for mass in mas_array:
        if mass == 1.00784:
            species.append(1)
        elif mass == 12.011:
            species.append(6)
        elif mass == 14.007:
            species.append(7)
        elif mass == 15.999:
            species.append(8)
        elif mass == 32.06:
            species.append(16)
        elif mass == 35.45:
            species.append(17)
        else:
            sys.exit("Not Implemented error.")

    return species


if __name__ == '__main__':
    #  0. Basic setup
    ## MODEL PARAMETERS + VAC MODEL + ENERGY FUNCTION
    # date = '290124'
    # date = '040324'
    date = '250424'
    compute_energies_of_precomputed_trajectories = True
    dataset_type = 'all'
    model_type = 'U_wat'
    postprocess_compute_energies = True

    if compute_energies_of_precomputed_trajectories:
        box = jnp.eye(3) * 1000
        r_cut = 4.0
        n_species = 100
        dr_thresh = 0.05
        neighbor_capacity_multiple = 2.7  # Hard coded for ANI1-x dataset.

        if model_type == 'U_vac':
            vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                             'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                             'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                             'Params.pkl'
            with open(vac_model_path, 'rb') as pickle_file:
                loaded_params = pickle.load(pickle_file)
        elif model_type == 'U_wat':
            t_prod_U_vac = 250
            t_equil_U_vac = 50
            num_epochs = 500
            print_every_time = 5
            initial_lr = 0.000001
            lr_decay = 0.1
            #save_checkpoint = ("020224_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(
            #    initial_lr) + "_lrd" + str(lr_decay))
            # save_checkpoint = ("040324_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(
            #     initial_lr) + "_lrd" + str(lr_decay))
            save_checkpoint = (date + "_t_prod_" + str(t_prod_U_vac) + "ps_t_equil_" +
                               str(t_equil_U_vac) + "ps_iL" + str(initial_lr) + "_lrd" +
                               str(lr_decay) + "_epochs" + str(num_epochs)
                               + "_seed7_train_389" + "mem_0.97")
            trained_params_path = 'checkpoints/' + save_checkpoint + f'_epoch499.pkl'
            with open(trained_params_path, 'rb') as pickle_file:
                loaded_params = pickle.load(pickle_file)

        #  1. Iterate all saved trajectories
        # data_num_list = get_404_data()
        # data_num_list = get_train_dataset()
        data_num_list = get_train_dataset_389()
        if dataset_type == 'test':
            # ind_list = onp.arange(514)
            ind_list = onp.arange(559)
            ind_list = onp.delete(ind_list, data_num_list)
            data_num_list = ind_list
        elif dataset_type == 'all':
            data_num_list = onp.arange(559)
        elif dataset_type == 'train':
            pass


        data_energy_pred = []
        counter = 0
        for data_num in data_num_list:
            counter += 1

            if model_type == 'U_vac':
                load_traj_path = 'precomputed_trajectories/250_50ps_load_traj_mol_' + str(data_num+1) + '_AC'
            elif model_type == 'U_wat':
                load_traj_path = 'precomputed_trajectories/250_50ps_load_wat_traj_mol_' + str(data_num+1) + '_AC'
            # if os.path.exists(load_traj_path):
            #     pass
            # else:
            #     print(str(data_num) + ' does not exist')
            with open(load_traj_path, 'rb') as file:
                traj_dict = dill.load(file)
            over_flow = traj_dict['over_flow']
            sim_state = traj_dict['sim_state']
            trajectory = traj_dict['trajectory']
            print(data_num)

            # Initialize nbrs
            config = neural_networks.initialize_nequip_cfg_MaxSetup(n_species, r_cut)
            displacement, shift_fn = space.periodic_general(box, fractional_coordinates=True)

            ## NEQUIP FUNCTIONS
            neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
                displacement, box, config, atoms=None, dr_threshold=dr_thresh,
                capacity_multiplier=neighbor_capacity_multiple,
                fractional_coordinates=True,
                disable_cell_list=True)

            all_pos = trajectory.position
            all_mass = trajectory.mass
            init_pos = all_pos[0]
            nbrs = neighbor_fn.allocate(init_pos, extra_capacity=0)
            energy_fn_init = energy_fn_template(loaded_params)

            @jit
            def compute_energy_and_forces_update_nbrs(pos, nbrs, species):
                nbrs = neighbor_fn.update(pos, nbrs)
                pred_energies, neg_forces = value_and_grad(energy_fn_init)(pos, neighbor=nbrs,
                                                                           species=species)
                return pred_energies, neg_forces, nbrs


            for count, pos in enumerate(all_pos):
                species = jnp.array(species_from_mass(all_mass[count]))
                energy_pred, _, nbrs = compute_energy_and_forces_update_nbrs(all_pos[count], nbrs, species)
                print("Energy: ", energy_pred)
                data_energy_pred.append(energy_pred)
            print("Done with molecule nr: " + str(counter) + ' with data_num: ' + str(data_num) + ' and final energy: ' +
                  str(energy_pred))

        if model_type == 'U_vac':
            onp.save('postprocessTrajectories/' + str(date) + 'dataset_type_' + dataset_type + '_250_50_ps_precomputedTrajectories_energies.npy', onp.array(data_energy_pred))
            x = onp.arange(len(data_energy_pred))
            plt.plot(x, data_energy_pred, color='black', linestyle='-')
            plt.xlabel("Sampled steps per molecule")
            plt.ylabel("Pred Energy")
            plt.savefig("postprocessTrajectories/" + str(date) + 'dataset_type_' + dataset_type + "_250_50_precomputedTrajectories.png")
        elif model_type == 'U_wat':
            onp.save('postprocessTrajectories/' + str(date) + 'dataset_type_' + dataset_type + '_250_50_precomputedTrajectories_Uwat_energies.npy', onp.array(data_energy_pred))
            x = onp.arange(len(data_energy_pred))
            plt.plot(x, data_energy_pred, color='black', linestyle='-')
            plt.xlabel("Sampled steps per molecule")
            plt.ylabel("Pred Energy")
            plt.savefig("postprocessTrajectories/" + str(date) + 'dataset_type_' + dataset_type + "_250_50_precomputedTrajectories_Uwat.png")
        # plt.show()


    if postprocess_compute_energies:

        if model_type == 'U_vac':
            energies = onp.load('postprocessTrajectories/' + str(date) + 'dataset_type_' + dataset_type + '_250_50_ps_precomputedTrajectories_energies.npy')
        elif model_type == 'U_wat':
            energies = onp.load('postprocessTrajectories/' + str(date) + 'dataset_type_' + dataset_type + '_250_50_precomputedTrajectories_Uwat_energies.npy')
        energies = energies.reshape(-1, 40)
        # energies = energies.reshape(-1, 200)
        # check if energies contains nan or 0
        print("Max energy: ", onp.max(energies))
        print("Min energy: ", onp.min(energies))
        std_energies = onp.std(energies, axis=1)
        print("Max std: ", onp.max(std_energies))
        energy_new = []
        for mol in range(energies.shape[0]):
            energy_new.append(energies[mol][0])
            energy_new.append(energies[mol][-1])

        # Plot energy_new 100 x values per plot
        len_energies = len(energy_new)
        energy_entries_per_plot = list(range(0, len_energies, 100))
        if len_energies % 100 != 0:
            energy_entries_per_plot.append(len_energies)
        for i in range(len(energy_entries_per_plot) - 1):
            x = onp.arange(energy_entries_per_plot[i], energy_entries_per_plot[i+1])
            plt.plot(x, energy_new[energy_entries_per_plot[i]:energy_entries_per_plot[i+1]], color='black', linestyle='-')
            plt.xlabel("50 molecules")
            plt.ylabel("Pred Energy")
            plt.show()



