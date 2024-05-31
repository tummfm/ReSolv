"""Script to postprocess the free energy predictions with exclusively BAR estimator (quicker than TI)"""
import os
import sys
import gc
#
#
# if len(sys.argv) > 1:
#     visible_device = str(sys.argv[1])
# else:
#     visible_device = 0


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from jax import config
config.update('jax_enable_x64', True)
## TRAINING UTILS NOTABLY CONTAINS A FUNCTION TO CREATE THE DATA SET
## I.E. CONFIGURATIONS, MASSES, SPECIES ETC. USED FOR THE TRAINING
import warnings
warnings.filterwarnings("ignore")

from training_utils_HFE import *

import numpy as onp
import jax.numpy as jnp
import optax
import pickle
from jax_md import space
import jax
import networkx as nx

## MODIFIED trainers_HFE.py module
from chemtrain import trainers, traj_util
from chemtrain.jax_md_mod import custom_quantity
from util import Postprocessing, Initialization
## LOADING BAR ESTIMATOR FUNCTION
from chemtrain.reweighting import init_bar
import time
from jax.lib import xla_bridge

print('Jax Device: ', xla_bridge.get_backend().platform)

## DOUBLE PRECISION
from jax import config
config.update('jax_enable_x64', True)
compute_debug_forces = False
evaluate_debug_forces = False
check_graphs = True
check_graphs_trajs = False
use_Uvac = False
use_Uwat = True
check_trajs_len = False


if check_trajs_len:
    load_trajs = True


    ## SYSTEM PARAMETERS
    system_temperature = 298.15  # Kelvin
    boltzmann_constant = 0.00198720426 # in kcal / mol K
    kbt = system_temperature * boltzmann_constant

    ## MODEL PARAMETERS
    ## USE MODEL 'Nequip_HFE' FOR THE HFE TRAINING W/O PRIOR
    model = 'Nequip_HFE'
    integrator = 'Langevin'
    kbt_dependent = False
    dropout_init_seed = None  # if no dropout should be used
    print(f'model : {model}')

    t_prod = 51
    t_equil = 1
    print_every_time = 0.1
    initial_lr = 0.000001
    lr_decay = 0.1
    num_epochs = 500

    date = "080524"
    t_prod_U_vac = 250
    t_equil_U_vac = 50
    data_type = "all"
    use_Uwat = False

    time_conversion = round(10**3 / 48.8882129,4) # 1 unit --> 1 ps
    time_step = 0.001 * time_conversion  # in ps
    total_time = t_prod * time_conversion  # in ps - 20
    t_equilib = t_equil * time_conversion  # in ps - 5
    print_every = print_every_time * time_conversion  # in ps - 0.1s


    vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                         'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                         'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                         'Params.pkl'


    train_dataset_389 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176, 181, 187, 188, 205, 206,
     211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 355, 356, 358, 360, 364, 1, 3, 6, 13, 18, 23, 49,
     63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233, 248, 259, 260, 272, 277, 280, 290,
     310, 311, 318, 337, 362, 363, 366, 377, 385, 22, 112, 173, 195, 210, 285, 302, 347, 372, 382, 390, 20, 123, 184, 234,
     268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50, 51, 52, 57, 59, 60, 61, 65, 66, 67, 70, 72,
     76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107, 109, 110, 116, 117, 120, 121, 127, 128, 131, 133, 135, 136, 137,
     140, 141, 146, 149, 156, 157, 161, 165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203, 208, 209, 213, 214, 218,
     221, 228, 235, 236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273, 274, 275, 276, 279, 284,
     287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343, 348, 352, 353, 354, 357,
     359, 365, 367, 368, 371, 375, 376, 381, 384, 388, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101, 103, 108, 111, 115, 119,
     129, 132, 134, 155, 159, 170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288, 295, 300, 321, 325, 330,
     342, 349, 379, 386, 407, 414, 0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293, 294, 29, 186, 323, 30,
     166, 190, 198, 207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96, 113, 118, 124, 125, 138,
     142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230, 231, 232, 239, 244, 246,
     253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335, 344, 345, 346, 351, 361, 369, 373, 374, 378,
     383, 392, 396, 380, 397, 404, 465, 513]

    # dataset_for_failed_molecules = [382, 196, 549, 294, 106, 395, 112, 210, 243, 147, 310, 91, 380, 126, 479]

    # dataset_for_failed_molecules = [91, 106, 112, 126, 147, 150, 196, 210, 243, 302, 310, 380,
    #                                 382, 395, 422, 456, 458, 479, 549]

    dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549,
                                    420]

    dataset_check = [37, 276]

    dataset_check = [106, 147, 196, 243, 380]

    exclusion_list = dataset_for_failed_molecules
    no_confs = 1

    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset, no_confs, exclusion_list, data_type='test')
    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    # data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(dataset_check, no_confs, None, data_type='train')


    ## THE DATA LIST IS STRUCTURED AS FOLLOWS:
    ## data = [r_init, mass, species, target_dict,smile_list[k], mol_{k+1}_AC']
    ## WHERE K REFERS TO THE KTH MOLECULE IN THE DATA SET


    ## DEFINING BOX EXAMPLE
    box = jnp.eye(3)*1000

    displacement, shift = space.periodic_general(
        box, fractional_coordinates=True)


    ## INITIALISING THE TIMINGS
    timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                          print_every)

    ## PREDICTIONS FROM MULTIPLE TRAJECTORIES

     ## LOADING TRAJECTORIES
    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0


    force_entries = {}

    # data = data[int_begin:int_end]
    unconnected_mols = []
    connected_mols = []
    traj_lens = []
    for r_init, mass, species, target_dict, smile, Mol in data:
        ## NEED PRED VALUE, EXP VALUE, EXP ERROR FOR ANALYSIS - GAFF PRED FOR COMPARISON AND SMILE COULD BE USEFUL TO KNOW
        ## PRED IS LIST AS WE ESTIMATE SEVERAL
        # print("Counter: ", counter)
        pred_dict[Mol] = {'exp': FE_values[counter], 'exp_err': errors[counter],
                          'GAFF_pred': GAFF_preds[counter], 'smile': smile}
        # print(Mol)
        # print(smile)

        simulation_data = Initialization.InitializationClass(r_init=r_init, box=box, kbt=kbt, masses=mass,
                                                             dt=time_step, species=species)

        reference_state, init_params, simulation_fns, compute_fns, targets = \
            Initialization.initialize_simulation(simulation_data, model, target_dict, integrator=integrator,
                                                     kbt_dependent=kbt_dependent, dropout_init_seed=dropout_init_seed,
                                                     load_trajectories=load_trajs, vac_model_path=vac_model_path)

        simulator_template, energy_fn_template, neighbor_fn = simulation_fns

        with open(f"precomputed_trajectories/250_50ps_load_traj_{Mol}", "rb") as file:
            traj_dict = pickle.load(file)

        aux = traj_dict['aux']
        barostat_press = traj_dict['barostat_press']
        ds = traj_dict['ds']
        df = traj_dict['df']
        over_flow = traj_dict['over_flow']
        sim_state = traj_dict['sim_state']
        thermostat_kbt = traj_dict['thermostat_kbt']
        trajectory = traj_dict['trajectory']
        pos = sim_state.position
        nbr = reference_state
        nbr = nbr.update(pos)
        sim_state = sim_state, nbr
        pos_Uvac = pos
        nbrs_Uvac = nbr

        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uvac_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=init_params, aux=aux)

        traj_lens.append(trajectory.position.shape[0])
        counter += 1
        print("counter ", counter)
print("debug")

if check_graphs_trajs:
    load_trajs = True


    ## SYSTEM PARAMETERS
    system_temperature = 298.15  # Kelvin
    boltzmann_constant = 0.00198720426 # in kcal / mol K
    kbt = system_temperature * boltzmann_constant

    ## MODEL PARAMETERS
    ## USE MODEL 'Nequip_HFE' FOR THE HFE TRAINING W/O PRIOR
    model = 'Nequip_HFE'
    integrator = 'Langevin'
    kbt_dependent = False
    dropout_init_seed = None  # if no dropout should be used
    print(f'model : {model}')

    t_prod = 51
    t_equil = 1
    print_every_time = 0.1
    initial_lr = 0.000001
    lr_decay = 0.1
    num_epochs = 500

    date = "080524"
    t_prod_U_vac = 250
    t_equil_U_vac = 50
    data_type = "all"

    time_conversion = round(10**3 / 48.8882129,4) # 1 unit --> 1 ps
    time_step = 0.001 * time_conversion  # in ps
    total_time = t_prod * time_conversion  # in ps - 20
    t_equilib = t_equil * time_conversion  # in ps - 5
    print_every = print_every_time * time_conversion  # in ps - 0.1s


    vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                         'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                         'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                         'Params.pkl'


    train_dataset_389 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176, 181, 187, 188, 205, 206,
     211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 355, 356, 358, 360, 364, 1, 3, 6, 13, 18, 23, 49,
     63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233, 248, 259, 260, 272, 277, 280, 290,
     310, 311, 318, 337, 362, 363, 366, 377, 385, 22, 112, 173, 195, 210, 285, 302, 347, 372, 382, 390, 20, 123, 184, 234,
     268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50, 51, 52, 57, 59, 60, 61, 65, 66, 67, 70, 72,
     76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107, 109, 110, 116, 117, 120, 121, 127, 128, 131, 133, 135, 136, 137,
     140, 141, 146, 149, 156, 157, 161, 165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203, 208, 209, 213, 214, 218,
     221, 228, 235, 236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273, 274, 275, 276, 279, 284,
     287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343, 348, 352, 353, 354, 357,
     359, 365, 367, 368, 371, 375, 376, 381, 384, 388, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101, 103, 108, 111, 115, 119,
     129, 132, 134, 155, 159, 170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288, 295, 300, 321, 325, 330,
     342, 349, 379, 386, 407, 414, 0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293, 294, 29, 186, 323, 30,
     166, 190, 198, 207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96, 113, 118, 124, 125, 138,
     142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230, 231, 232, 239, 244, 246,
     253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335, 344, 345, 346, 351, 361, 369, 373, 374, 378,
     383, 392, 396, 380, 397, 404, 465, 513]

    # dataset_for_failed_molecules = [382, 196, 549, 294, 106, 395, 112, 210, 243, 147, 310, 91, 380, 126, 479]

    # dataset_for_failed_molecules = [91, 106, 112, 126, 147, 150, 196, 210, 243, 302, 310, 380,
    #                                 382, 395, 422, 456, 458, 479, 549]

    # dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549,
    #                                 420]

    dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]
    dataset_for_failed_U_wat = [30, 190, 280, 294, 420]

    exclusion_list = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549, 30, 190, 280, 294, 420]

    exclusion_list = dataset_for_failed_molecules + dataset_for_failed_U_wat
    no_confs = 1

    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset, no_confs, exclusion_list, data_type='test')
    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    # data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type='all')


    ## THE DATA LIST IS STRUCTURED AS FOLLOWS:
    ## data = [r_init, mass, species, target_dict,smile_list[k], mol_{k+1}_AC']
    ## WHERE K REFERS TO THE KTH MOLECULE IN THE DATA SET


    ## DEFINING BOX EXAMPLE
    box = jnp.eye(3)*1000

    displacement, shift = space.periodic_general(
        box, fractional_coordinates=True)


    ## INITIALISING THE TIMINGS
    timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                          print_every)

    ## PREDICTIONS FROM MULTIPLE TRAJECTORIES

     ## LOADING TRAJECTORIES
    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0


    force_entries = {}

    # data = data[int_begin:int_end]
    unconnected_mols = []
    connected_mols = []
    for r_init, mass, species, target_dict, smile, Mol in data:
        ## NEED PRED VALUE, EXP VALUE, EXP ERROR FOR ANALYSIS - GAFF PRED FOR COMPARISON AND SMILE COULD BE USEFUL TO KNOW
        ## PRED IS LIST AS WE ESTIMATE SEVERAL
        # print("Counter: ", counter)
        pred_dict[Mol] = {'exp': FE_values[counter], 'exp_err': errors[counter],
                          'GAFF_pred': GAFF_preds[counter], 'smile': smile}
        # print(Mol)
        # print(smile)

        simulation_data = Initialization.InitializationClass(r_init=r_init, box=box, kbt=kbt, masses=mass,
                                                             dt=time_step, species=species)

        reference_state, init_params, simulation_fns, compute_fns, targets = \
            Initialization.initialize_simulation(simulation_data, model, target_dict, integrator=integrator,
                                                     kbt_dependent=kbt_dependent, dropout_init_seed=dropout_init_seed,
                                                     load_trajectories=load_trajs, vac_model_path=vac_model_path)

        simulator_template, energy_fn_template, neighbor_fn = simulation_fns

        if use_Uvac:
            with open(f"precomputed_trajectories/250_50ps_load_traj_{Mol}", "rb") as file:
                traj_dict = pickle.load(file)
        elif use_Uwat:
            with open(f"precomputed_trajectories/250_50ps_load_wat_traj_{Mol}", "rb") as file:
                traj_dict = pickle.load(file)
        else:
            sys.exit("Neither Uvac nor Uwat chosen.")

        aux = traj_dict['aux']
        barostat_press = traj_dict['barostat_press']
        ds = traj_dict['ds']
        df = traj_dict['df']
        over_flow = traj_dict['over_flow']
        sim_state = traj_dict['sim_state']
        thermostat_kbt = traj_dict['thermostat_kbt']
        trajectory = traj_dict['trajectory']
        pos = sim_state.position
        nbr = reference_state
        nbr = nbr.update(pos)
        sim_state = sim_state, nbr
        pos_Uvac = pos
        nbrs_Uvac = nbr

        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uvac_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=init_params, aux=aux)


        # check first time step or trajectory
        check_first_time_step = False
        if check_first_time_step:
            print("0.1 ps")
            nbrs_Uvac = nbrs_Uvac.update(trajectory.position[0])

            # Use nbrs to generate graphx - check this for entire trajectory and find time it fails
            G = nx.Graph()
            edges_precomputed = [(int(nbrs_Uvac.idx[0][count]), int(nbrs_Uvac.idx[1][count])) for count in
                                 range(onp.sum(nbrs_Uvac.idx[0] != nbrs_Uvac.reference_position.shape[0]))]
            G.add_nodes_from(onp.arange(nbrs_Uvac.reference_position.shape[0]))
            G.add_edges_from(edges_precomputed)
            check_if_one_graph = nx.is_connected(G)

            if not check_if_one_graph:
                print("Uvac disconnected: {}".format(Mol))
                unconnected_mols.append((Mol, '0.1 ps'))
                break

        else:
            # Iterate over time steps
            for sample in range(trajectory.position.shape[0]):
                print("{} ps".format(sample*5 + 5))
                nbrs_Uvac = nbrs_Uvac.update(trajectory.position[sample])

                # Use nbrs to generate graphx - check this for entire trajectory and find time it fails
                G = nx.Graph()
                edges_precomputed = [(int(nbrs_Uvac.idx[0][count]), int(nbrs_Uvac.idx[1][count])) for count in range(onp.sum(nbrs_Uvac.idx[0] != nbrs_Uvac.reference_position.shape[0]))]
                G.add_nodes_from(onp.arange(nbrs_Uvac.reference_position.shape[0]))
                G.add_edges_from(edges_precomputed)
                check_if_one_graph = nx.is_connected(G)

                if not check_if_one_graph:
                    print("Uvac disconnected: {}".format(Mol))
                    unconnected_mols.append(Mol)
                    break

        counter += 1
    print("unconnected mols: ", unconnected_mols)
if check_graphs:
    # int_begin = int(sys.argv[1])
    # int_end = int(sys.argv[2])
    # int_begin = 0
    # int_end = 17
    ## LOAD PRECOMPUTED TRAJECTORIES OR NOT
    load_trajs = True


    ## SYSTEM PARAMETERS
    system_temperature = 298.15  # Kelvin
    boltzmann_constant = 0.00198720426 # in kcal / mol K
    kbt = system_temperature * boltzmann_constant

    ## MODEL PARAMETERS
    ## USE MODEL 'Nequip_HFE' FOR THE HFE TRAINING W/O PRIOR
    model = 'Nequip_HFE'
    integrator = 'Langevin'
    kbt_dependent = False
    dropout_init_seed = None  # if no dropout should be used
    print(f'model : {model}')

    t_prod = 250
    t_equil = 50
    print_every_time = 5
    initial_lr = 0.000001
    lr_decay = 0.1
    num_epochs = 500

    date = "080524"
    t_prod_U_vac = 250
    t_equil_U_vac = 50
    data_type = "all"
    use_Uwat = True
    use_Uvac = False

    time_conversion = round(10**3 / 48.8882129,4) # 1 unit --> 1 ps
    time_step = 0.001 * time_conversion  # in ps
    total_time = t_prod * time_conversion  # in ps - 20
    t_equilib = t_equil * time_conversion  # in ps - 5
    print_every = print_every_time * time_conversion  # in ps - 0.1s


    vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                         'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                         'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                         'Params.pkl'
    save_checkpoint = (date + "_t_prod_" + str(t_prod_U_vac) + "ps_t_equil_" +
                       str(t_equil_U_vac) + "ps_iL" + str(initial_lr) + "_lrd" +
                       str(lr_decay) + "_epochs" + str(num_epochs)
                       + "_seed7_train_389" +
                       "mem_0.97")
    trained_params_path = 'checkpoints/' + save_checkpoint + '_epoch499.pkl'



    # Load or compute trajs
    # save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
    #                   'dataset_' + data_type + '.pkl')


    train_dataset_389 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176, 181, 187, 188, 205, 206,
     211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 355, 356, 358, 360, 364, 1, 3, 6, 13, 18, 23, 49,
     63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233, 248, 259, 260, 272, 277, 280, 290,
     310, 311, 318, 337, 362, 363, 366, 377, 385, 22, 112, 173, 195, 210, 285, 302, 347, 372, 382, 390, 20, 123, 184, 234,
     268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50, 51, 52, 57, 59, 60, 61, 65, 66, 67, 70, 72,
     76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107, 109, 110, 116, 117, 120, 121, 127, 128, 131, 133, 135, 136, 137,
     140, 141, 146, 149, 156, 157, 161, 165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203, 208, 209, 213, 214, 218,
     221, 228, 235, 236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273, 274, 275, 276, 279, 284,
     287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343, 348, 352, 353, 354, 357,
     359, 365, 367, 368, 371, 375, 376, 381, 384, 388, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101, 103, 108, 111, 115, 119,
     129, 132, 134, 155, 159, 170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288, 295, 300, 321, 325, 330,
     342, 349, 379, 386, 407, 414, 0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293, 294, 29, 186, 323, 30,
     166, 190, 198, 207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96, 113, 118, 124, 125, 138,
     142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230, 231, 232, 239, 244, 246,
     253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335, 344, 345, 346, 351, 361, 369, 373, 374, 378,
     383, 392, 396, 380, 397, 404, 465, 513]

    # dataset_for_failed_molecules = [382, 196, 549, 294, 106, 395, 112, 210, 243, 147, 310, 91, 380, 126, 479]

    # rerun_after_50ps_worked = [106, 147, 196, 243, 380]

    dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]
    dataset_for_failed_U_wat = [30, 190, 280, 294, 420]

    exclusion_list = dataset_for_failed_molecules
    no_confs = 1

    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset, no_confs, exclusion_list, data_type='test')
    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    # data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    # data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type='all')
    # data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(rerun_after_50ps_worked, no_confs,
    #                                                                                      exclusion_list,
    #                                                                                      data_type='train')

    data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(dataset_for_failed_molecules,
                                                                                         no_confs,
                                                                                         exclusion_list,
                                                                                         data_type='all')


    ## THE DATA LIST IS STRUCTURED AS FOLLOWS:
    ## data = [r_init, mass, species, target_dict,smile_list[k], mol_{k+1}_AC']
    ## WHERE K REFERS TO THE KTH MOLECULE IN THE DATA SET


    ## DEFINING BOX EXAMPLE
    box = jnp.eye(3)*1000

    displacement, shift = space.periodic_general(
        box, fractional_coordinates=True)


    ## INITIALISING THE TIMINGS
    timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                          print_every)

    # LOADING TRAINED PARAMETERS
    with open(trained_params_path, 'rb') as pickle_file:
        loaded_params = pickle.load(pickle_file)

    ## PREDICTIONS FROM MULTIPLE TRAJECTORIES

     ## LOADING TRAJECTORIES
    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0


    force_entries = {}

    # data = data[int_begin:int_end]
    unconnected_mols = []
    for r_init, mass, species, target_dict, smile, Mol in data:
        print("Counter: ", counter)
        ## NEED PRED VALUE, EXP VALUE, EXP ERROR FOR ANALYSIS - GAFF PRED FOR COMPARISON AND SMILE COULD BE USEFUL TO KNOW
        ## PRED IS LIST AS WE ESTIMATE SEVERAL
        # print("Counter: ", counter)
        pred_dict[Mol] = {'exp': FE_values[counter], 'exp_err': errors[counter],
                          'GAFF_pred': GAFF_preds[counter], 'smile': smile}
        # print(Mol)
        # print(smile)

        simulation_data = Initialization.InitializationClass(r_init=r_init, box=box, kbt=kbt, masses=mass,
                                                             dt=time_step, species=species)

        reference_state, init_params, simulation_fns, compute_fns, targets = \
            Initialization.initialize_simulation(simulation_data, model, target_dict, integrator=integrator,
                                                     kbt_dependent=kbt_dependent, dropout_init_seed=dropout_init_seed,
                                                     load_trajectories=load_trajs, vac_model_path=vac_model_path)

        simulator_template, energy_fn_template, neighbor_fn = simulation_fns

        ## LOAD TRAJECTORIES E.G
        ## VAC
        # with open(f'{loading_traj_path}_{Mol}_vac.pkl', 'rb') as f:
        #     traj_dict = pickle.load(f)
        # with open(f"precomputed_trajectories/220ps_load_traj_{Mol}", "rb") as file:
        #     traj_dict = pickle.load(file)

        if use_Uvac:
            with open(f"precomputed_trajectories/250_50ps_load_traj_{Mol}", "rb") as file:
                traj_dict = pickle.load(file)


            aux = traj_dict['aux']
            barostat_press = traj_dict['barostat_press']
            ds = traj_dict['ds']
            df = traj_dict['df']
            over_flow = traj_dict['over_flow']
            sim_state = traj_dict['sim_state']
            thermostat_kbt = traj_dict['thermostat_kbt']
            trajectory = traj_dict['trajectory']
            pos = sim_state.position
            nbr = reference_state
            nbr = nbr.update(pos)
            sim_state = sim_state, nbr
            pos_Uvac = pos
            nbrs_Uvac = nbr

            ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
            Uvac_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                     overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                     barostat_press=barostat_press, entropy_diff=ds,
                                                     free_energy_diff=df, energy_params=init_params, aux=aux)

            # Use nbrs to generate graphx
            G = nx.Graph()
            edges_precomputed = [(int(nbrs_Uvac.idx[0][count]), int(nbrs_Uvac.idx[1][count])) for count in range(onp.sum(nbrs_Uvac.idx[0] != nbrs_Uvac.reference_position.shape[0]))]
            G.add_nodes_from(onp.arange(nbrs_Uvac.reference_position.shape[0]))
            G.add_edges_from(edges_precomputed)
            check_if_one_graph = nx.is_connected(G)

            if not check_if_one_graph:
                print("Uvac disconnected: {}".format(Mol))
                unconnected_mols.append(Mol)
            # else:
            #     energy_fn_Uvac = energy_fn_template(init_params)
            #     _, pred_force_Uvac = jax.value_and_grad(energy_fn_Uvac)(pos_Uvac, nbrs_Uvac, species)
            #     print(pred_force_Uvac)
            #     # all_distances = [onp.linalg.norm(displacement(i, j)) for count_1, i in
            #     #                  enumerate(onp.array(pos_Uvac)) for count_2, j in
            #     #                  enumerate(onp.array(pos_Uvac)) if count_1 != count_2]
            #     # max_dist = onp.max(all_distances)



        elif use_Uwat:
            ## WAT
            with open(f"precomputed_trajectories/250_50ps_load_wat_traj_{Mol}", "rb") as file:
                traj_dict = pickle.load(file)

            aux = traj_dict['aux']
            barostat_press = traj_dict['barostat_press']
            ds = traj_dict['ds']
            df = traj_dict['df']
            over_flow = traj_dict['over_flow']
            sim_state = traj_dict['sim_state']
            thermostat_kbt = traj_dict['thermostat_kbt']
            trajectory = traj_dict['trajectory']
            pos = sim_state.position
            nbr = reference_state
            nbr = nbr.update(pos)
            sim_state = sim_state, nbr

            pos_Uwat = pos
            nbrs_Uwat = nbr
            ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
            Uwat_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                     overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                     barostat_press=barostat_press, entropy_diff=ds,
                                                     free_energy_diff=df, energy_params=loaded_params, aux=aux)

            G = nx.Graph()
            edges_precomputed = [(int(nbrs_Uwat.idx[0][count]), int(nbrs_Uwat.idx[1][count])) for count in
                                 range(onp.sum(nbrs_Uwat.idx[0] != nbrs_Uwat.reference_position.shape[0]))]
            G.add_nodes_from(onp.arange(nbrs_Uwat.reference_position.shape[0]))
            G.add_edges_from(edges_precomputed)
            check_if_one_graph = nx.is_connected(G)

            if not check_if_one_graph:
                print("Uvac disconnected: {}".format(Mol))
                unconnected_mols.append(Mol)

        counter += 1
    else:
        print("Specify whether to evaluate Uvac or Uwat")

print("Unconnected mols: ", unconnected_mols)
print("Len mols: ", len(unconnected_mols))

if compute_debug_forces:
    # int_begin = int(sys.argv[1])
    # int_end = int(sys.argv[2])
    int_begin = 0
    int_end = 17
    ## LOAD PRECOMPUTED TRAJECTORIES OR NOT
    load_trajs = True


    ## SYSTEM PARAMETERS
    system_temperature = 298.15  # Kelvin
    boltzmann_constant = 0.00198720426 # in kcal / mol K
    kbt = system_temperature * boltzmann_constant

    ## MODEL PARAMETERS
    ## USE MODEL 'Nequip_HFE' FOR THE HFE TRAINING W/O PRIOR
    model = 'Nequip_HFE'
    integrator = 'Langevin'
    kbt_dependent = False
    dropout_init_seed = None  # if no dropout should be used
    print(f'model : {model}')

    ## TIME PARAMETERS
    ## TIME CONVERSION IS USED TO CONVERT THE TIME STEPS TO PS
    ## I.E. 1 * time_conversion = 1 ps
    ## USE ROUNDING TO AVOID FLOATING POINT ERRORS, DEPENDING ON THE TIME PARAMETERS THIS CAN BE ADJUSTED
    t_prod = 250
    t_equil = 50
    print_every_time = 5
    initial_lr = 0.000001
    lr_decay = 0.1
    # date = '040324'
    num_epochs = 500

    date = "250424"
    t_prod_U_vac = 250
    t_equil_U_vac = 50
    data_type = "all"
    use_Uwat = False

    time_conversion = round(10**3 / 48.8882129,4) # 1 unit --> 1 ps
    time_step = 0.001 * time_conversion  # in ps
    total_time = t_prod * time_conversion  # in ps - 20
    t_equilib = t_equil * time_conversion  # in ps - 5
    print_every = print_every_time * time_conversion  # in ps - 0.1s

    # vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
    #                  'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
    #                  'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
    #                  'Params.pkl'
    # save_checkpoint = ("040324_t_prod_" + str(t_prod) + "ps_t_equil_" + str(t_equil) + "ps_iL" + str(initial_lr) + "_lrd" + str(lr_decay))
    # trained_params_path = 'checkpoints/' + save_checkpoint + f'_epoch499.pkl'


    vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                         'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                         'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                         'Params.pkl'
    save_checkpoint = (date + "_t_prod_" + str(t_prod_U_vac) + "ps_t_equil_" +
                       str(t_equil_U_vac) + "ps_iL" + str(initial_lr) + "_lrd" +
                       str(lr_decay) + "_epochs" + str(num_epochs)
                       + "_seed7_train_389" +
                       "mem_0.97")
    trained_params_path = 'checkpoints/' + save_checkpoint + '_epoch499.pkl'



    # Load or compute trajs
    save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
                      'dataset_' + data_type + '.pkl')


    train_dataset_389 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176, 181, 187, 188, 205, 206,
     211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 355, 356, 358, 360, 364, 1, 3, 6, 13, 18, 23, 49,
     63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233, 248, 259, 260, 272, 277, 280, 290,
     310, 311, 318, 337, 362, 363, 366, 377, 385, 22, 112, 173, 195, 210, 285, 302, 347, 372, 382, 390, 20, 123, 184, 234,
     268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50, 51, 52, 57, 59, 60, 61, 65, 66, 67, 70, 72,
     76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107, 109, 110, 116, 117, 120, 121, 127, 128, 131, 133, 135, 136, 137,
     140, 141, 146, 149, 156, 157, 161, 165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203, 208, 209, 213, 214, 218,
     221, 228, 235, 236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273, 274, 275, 276, 279, 284,
     287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343, 348, 352, 353, 354, 357,
     359, 365, 367, 368, 371, 375, 376, 381, 384, 388, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101, 103, 108, 111, 115, 119,
     129, 132, 134, 155, 159, 170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288, 295, 300, 321, 325, 330,
     342, 349, 379, 386, 407, 414, 0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293, 294, 29, 186, 323, 30,
     166, 190, 198, 207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96, 113, 118, 124, 125, 138,
     142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230, 231, 232, 239, 244, 246,
     253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335, 344, 345, 346, 351, 361, 369, 373, 374, 378,
     383, 392, 396, 380, 397, 404, 465, 513]

    dataset_for_failed_molecules = [382, 196, 549, 294, 106, 395, 112, 210, 243, 147, 310, 91, 380, 126, 479]

    exclusion_list = None

    ## WE INTIALISE SEVERAL STATES SO THAT WE CAN ESTIMATE AN ERROR TO THE PREDICTIONS
    if not load_trajs:
        no_confs = 10
    else:
        no_confs = 1

    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset, no_confs, exclusion_list, data_type='test')
    # data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    # data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
    data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(dataset_for_failed_molecules, no_confs, exclusion_list, data_type='train')


    ## THE DATA LIST IS STRUCTURED AS FOLLOWS:
    ## data = [r_init, mass, species, target_dict,smile_list[k], mol_{k+1}_AC']
    ## WHERE K REFERS TO THE KTH MOLECULE IN THE DATA SET


    ## DEFINING BOX EXAMPLE
    box = jnp.eye(3)*1000

    displacement, shift = space.periodic_general(
        box, fractional_coordinates=True)


    ## INITIALISING THE TIMINGS
    timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                          print_every)

    # LOADING TRAINED PARAMETERS
    with open(trained_params_path, 'rb') as pickle_file:
        loaded_params = pickle.load(pickle_file)

    ## PREDICTIONS FROM MULTIPLE TRAJECTORIES

     ## LOADING TRAJECTORIES
    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0


    force_entries = {}

    data = data[int_begin:int_end]
    for r_init, mass, species, target_dict, smile, Mol in data:
        ## NEED PRED VALUE, EXP VALUE, EXP ERROR FOR ANALYSIS - GAFF PRED FOR COMPARISON AND SMILE COULD BE USEFUL TO KNOW
        ## PRED IS LIST AS WE ESTIMATE SEVERAL
        print("Counter: ", counter)
        pred_dict[Mol] = {'exp': FE_values[counter], 'exp_err': errors[counter],
                          'GAFF_pred': GAFF_preds[counter], 'smile': smile}
        print(Mol)
        print(smile)

        simulation_data = Initialization.InitializationClass(r_init=r_init, box=box, kbt=kbt, masses=mass,
                                                             dt=time_step, species=species)

        reference_state, init_params, simulation_fns, compute_fns, targets = \
            Initialization.initialize_simulation(simulation_data, model, target_dict, integrator=integrator,
                                                     kbt_dependent=kbt_dependent, dropout_init_seed=dropout_init_seed,
                                                     load_trajectories=load_trajs, vac_model_path=vac_model_path)

        simulator_template, energy_fn_template, neighbor_fn = simulation_fns

        ## LOAD TRAJECTORIES E.G
        ## VAC
        # with open(f'{loading_traj_path}_{Mol}_vac.pkl', 'rb') as f:
        #     traj_dict = pickle.load(f)
        # with open(f"precomputed_trajectories/220ps_load_traj_{Mol}", "rb") as file:
        #     traj_dict = pickle.load(file)
        with open(f"precomputed_trajectories/250_50ps_load_traj_{Mol}", "rb") as file:
            traj_dict = pickle.load(file)


        aux = traj_dict['aux']
        barostat_press = traj_dict['barostat_press']
        ds = traj_dict['ds']
        df = traj_dict['df']
        over_flow = traj_dict['over_flow']
        sim_state = traj_dict['sim_state']
        thermostat_kbt = traj_dict['thermostat_kbt']
        trajectory = traj_dict['trajectory']
        pos = sim_state.position
        nbr = reference_state
        nbr = nbr.update(pos)
        sim_state = sim_state, nbr
        pos_Uvac = pos
        nbrs_Uvac = nbr

        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uvac_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=init_params, aux=aux)

        if use_Uwat:
            ## WAT
            with open(f"precomputed_trajectories/250_50ps_load_wat_traj_{Mol}", "rb") as file:
                traj_dict = pickle.load(file)

            aux = traj_dict['aux']
            barostat_press = traj_dict['barostat_press']
            ds = traj_dict['ds']
            df = traj_dict['df']
            over_flow = traj_dict['over_flow']
            sim_state = traj_dict['sim_state']
            thermostat_kbt = traj_dict['thermostat_kbt']
            trajectory = traj_dict['trajectory']
            pos = sim_state.position
            nbr = reference_state
            nbr = nbr.update(pos)
            sim_state = sim_state, nbr

            pos_Uwat = pos
            nbrs_Uwat = nbr
            ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
            Uwat_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                     overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                     barostat_press=barostat_press, entropy_diff=ds,
                                                     free_energy_diff=df, energy_params=loaded_params, aux=aux)

            energy_fn_Uwat = energy_fn_template(loaded_params)
            energy_Uwat, pred_force_Uwat = jax.value_and_grad(energy_fn_Uwat)(pos_Uwat, nbrs_Uwat, species)

        energy_fn_Uvac = energy_fn_template(init_params)


        # energy takes pos, nbrs, species
        energy_Uvac, pred_force_Uvac = jax.value_and_grad(energy_fn_Uvac)(pos_Uvac, nbrs_Uvac, species)


        # force_Uvac, force_uwat = get_force(init_params, loaded_params, pos_Uvac, nbrs_Uvac, pos_Uwat, nbrs_Uwat, species)

        if use_Uwat:
            force_entries[Mol] = [pred_force_Uvac, pred_force_Uwat]
        else:
            force_entries[Mol] = [pred_force_Uvac]


        counter += 1

    if use_Uwat:
        with open('postprocessTrajectories/all_forces_debug_' + str(int_begin) + '_' + str(int_end) + '.pkl', 'wb') as pickle_file:
            pickle.dump(force_entries, pickle_file)
    else:
        with open('postprocessTrajectories/Uvac_forces_debug_' + str(int_begin) + '_' + str(int_end) + '.pkl', 'wb') as pickle_file:
            pickle.dump(force_entries, pickle_file)

if evaluate_debug_forces:
    force_dict = {}
    # int_begin_list = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
    #                   420, 440, 460, 480, 500, 520, 540]
    # int_end_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
    #                   420, 440, 460, 480, 500, 520, 540, 561]
    int_begin_list = [0]
    int_end_list = [17]
    for count in range(len(int_begin_list)):
        with open('postprocessTrajectories/Uvac_forces_debug_' + str(int_begin_list[count]) + '_' + str(int_end_list[count]) + '.pkl',
                  'rb') as pickle_file:
            temp_dict = pickle.load(pickle_file)
        force_dict.update(temp_dict)
    force_vac_list = []
    force_wat_list = []
    keys_list = []
    for key in force_dict.keys():
        force_vac_list.append(force_dict[key][0])
        # force_wat_list.append(force_dict[key][1])
        keys_list.append(key)

    print("Debug")
