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

## TRAINING UTILS NOTABLY CONTAINS A FUNCTION TO CREATE THE DATA SET
## I.E. CONFIGURATIONS, MASSES, SPECIES ETC. USED FOR THE TRAINING
from training_utils_HFE import *

import numpy as onp
import jax.numpy as jnp
import optax
import pickle
from jax_md import space
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

## LOAD PRECOMPUTED TRAJECTORIES OR NOT
load_trajs = True
check_convergence = False
# use_subset = int(sys.argv[1])
use_subset = 10
check_graph_connection = True

## EXAMPLE PATHS FOR SAVING PARAMS AND TO LOAD THE VACUUM PARAMS
# TODO
# vac_model_path='290623_QM7x_Nequip_Nequip_ID_QM7x_Prior_Precomp_34_Trainer_ANI1x_Energies4195237samples_20epochs_gu2emin7_gf1emin3_iL1emin2_LRdecay1emin4_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_13_bestParams.pkl'
# trained_params_path = 'some/path/to/params'
#save_pred_path = 'HFE_postprocess_resultspredictions.pkl'
# loading_traj_path = 'some/path/to/trajectories.pkl'


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

# date = "250424"
date = "080524"
t_prod_U_vac = 250
t_equil_U_vac = 50
# data_type = "test"
data_type = "test"

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



# # Load or compute trajs
# save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
#                   'dataset_' + data_type + '.pkl')


if check_convergence:
    save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
                      'dataset_' + data_type + '_use_subset_'+str(use_subset)+'.pkl')
else:
    save_pred_path = ('HFE_postprocess_results/' + save_checkpoint + '_HFE_predictions_load_trajs_' + str(load_trajs) +
                      'dataset_' + data_type + '.pkl')

_404 = [4, 23, 24, 34, 52, 55, 56, 65, 93, 112, 119, 131, 132, 137, 138, 143, 147, 149, 162, 168, 169, 184, 185,
        189, 193,202, 231, 242, 245, 270, 272, 274, 276, 278, 279, 295, 322, 326, 327, 329, 331, 335, 392, 393, 415, 424,
        425, 426,1, 3, 6, 13, 22, 48, 60, 66, 77, 82, 83, 87, 105, 133, 134, 139, 148, 177, 178, 190, 206, 209, 224, 235,
        236, 247,252, 255, 265, 283, 284, 291, 310, 333, 334, 347, 355, 387, 394, 400, 405, 407, 409, 418, 21, 103, 156, 176,
        188,260, 275, 319, 342, 352, 359, 364, 19, 113, 165, 210, 246, 244, 9, 11, 12, 18, 20, 25, 26, 31, 33, 35, 36,
        38, 39,41, 43, 44, 47, 49, 50, 51, 57, 58, 62, 63, 64, 67, 69, 72, 73, 74, 75, 86, 90, 91, 92, 94, 96, 97, 98, 100,
        101, 107, 108, 111, 116, 117, 120, 122, 124, 125, 126, 129, 136, 142, 146, 150, 152, 157, 158, 160, 161, 163,166, 174,
        175, 182, 186, 187, 191, 195, 198, 205, 211, 212, 214, 216, 218, 221, 223, 226, 227, 230, 232, 233, 238,
        239, 243,248, 249, 250, 251, 254, 259, 262, 264, 266, 271, 285, 286, 288, 290, 297, 301, 304, 305, 307, 309, 312, 313, 315,
        320, 324, 325, 328, 330, 336, 337, 338, 341, 345, 346, 351, 354, 358, 360, 362, 363, 368, 369, 370, 374,
        377, 378,380, 381, 383, 385, 388, 391, 396, 8, 42, 53, 59, 61, 71, 79, 81, 95, 99, 102, 106, 110, 118, 121, 123, 141,
        144,153, 154, 170, 180, 194, 197, 199, 201, 225, 228, 240, 253, 258, 263, 268, 273, 294, 298, 303, 314, 321,
        349, 356, 375, 382, 398, 401, 406, 427, 453, 0, 17, 30, 32, 37, 54, 80, 84, 128, 155, 213, 217, 219, 256, 267, 282,
        293, 311,28, 167, 296, 366, 29, 171, 350, 384, 14, 277, 413, 2, 5, 7, 10, 15, 16, 27, 40, 45, 46, 68, 70, 76, 78, 88,
        89, 104, 109, 114, 115, 127, 130, 135, 140, 145, 151, 159, 164, 172, 173, 179, 181, 183, 192, 196, 200, 203,
        204, 207, 208, 215, 220, 222, 229, 234, 237, 241, 261, 269, 280, 281, 287, 289, 292, 300, 302, 306, 308, 316, 317,
        318, 323,332, 339, 343, 344, 348, 353, 361, 365, 371, 373, 376, 379, 386, 390, 395, 397]

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

# exclusion_list = None

exclude_Uvac = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]
exclude_Uwat = [30, 190, 280, 294, 420]
exclusion_list = exclude_Uvac + exclude_Uwat

## WE INTIALISE SEVERAL STATES SO THAT WE CAN ESTIMATE AN ERROR TO THE PREDICTIONS
if not load_trajs:
    no_confs = 10
else:
    no_confs = 1

# data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset, no_confs, exclusion_list, data_type='test')
# data, FE_values, smiles, GAFF_preds, errors = generate_data(train_dataset_389, no_confs, exclusion_list, data_type=data_type)
data, FE_values, smiles, GAFF_preds, errors, mobley_list = generate_data_with_mobley(train_dataset_389, no_confs, exclusion_list, data_type=data_type)


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
if not load_trajs:

    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0
    for r_inits, mass, species, target_dict, smile, Mol in data:
        ## NEED PRED VALUE, EXP VALUE, EXP ERROR FOR ANALYSIS - GAFF PRED FOR COMPARISON AND SMILE COULD BE USEFUL TO KNOW
        ## PRED IS LIST AS WE ESTIMATE SEVERAL
        pred_dict[Mol] = {'predicitons':[], 'exp':FE_values[counter],'exp_err':errors[counter],'GAFF_pred': GAFF_preds[counter], 'smile':smile}
        print(Mol)
        print(smile)

        for r_init in r_inits:

            simulation_data = Initialization.InitializationClass(r_init=r_init, box=box, kbt=kbt, masses=mass, dt=time_step, species=species)

            # TODO - define loaded_params
            reference_state, init_params, simulation_fns, compute_fns, targets = \
                    Initialization.initialize_simulation(simulation_data, model, target_dict, integrator=integrator,
                                                         kbt_dependent=kbt_dependent,
                                                         dropout_init_seed=dropout_init_seed,
                                                         vac_model_path=vac_model_path, loaded_params=loaded_params)

            simulator_template, energy_fn_template, neighbor_fn = simulation_fns

            quantities = {}
            quantities['energy'] = custom_quantity.energy_wrapper(energy_fn_template)

            # INITIALISE TRAJECTORY GENERATOR
            trajectory_generator = jax.jit(traj_util.trajectory_generator_init_HFE(
                simulator_template, energy_fn_template, timings, quantities))

            # RETRIEVING SEPARATELY THE VACUUM AND WATER TRAJECTORIES
            reference_state_vac, reference_state_wat = reference_state

            ## UVAC TRAJ GENERATION - RUN MULTIPLE TIMES TO EQUILIABRATE
            print('Uvac traj generation')
            for m in range(3):
                print('traj',m)

                Uvac_traj = trajectory_generator(init_params, reference_state_vac, species,
                                             kT=kbt, pressure=None)
                reference_state_vac = Uvac_traj.sim_state

            ## UWAT TRAJ GENERATION - RUN MULTIPLE TIMES TO EQUILIABRATE
            print('Uwat traj generation')
            for m in range(3):
                print('traj',m)

                Uwat_traj = trajectory_generator(loaded_params, reference_state_wat, species,
                                             kT=kbt, pressure=None)
                reference_state_wat = Uwat_traj.sim_state

            ## BAR FREE ENERGY CALCULATION
            bennett_free_energy = init_bar(
                energy_fn_template, kbt, 10, 25)

            df,_ = bennett_free_energy(Uvac_traj, Uwat_traj, species)

            pred_dict[Mol]['predicitons'].append(df)

        counter += 1

elif (load_trajs and check_convergence):

    onp.random.seed(7)
    indices_list = onp.arange(len(data))
    onp.random.shuffle(indices_list)
    data = [data[i] for i in indices_list]
    FE_values = [FE_values[i] for i in indices_list]
    smiles = [smiles[i] for i in indices_list]
    GAFF_preds = [GAFF_preds[i] for i in indices_list]
    errors = [errors[i] for i in indices_list]

    data = data[:12]
    FE_values = FE_values[:12]
    smiles = smiles[:12]
    GAFF_preds = GAFF_preds[:12]
    errors = errors[:12]


    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0
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
        with open(f"precomputed_trajectories/4300_300ps_load_traj_{Mol}", "rb") as file:
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
        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uvac_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=init_params, aux=aux)
        ## WAT
        with open(f"precomputed_trajectories/4300_300ps_load_wat_traj_{Mol}", "rb") as file:
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
        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uwat_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=loaded_params, aux=aux)

        ## BAR FREE ENERGY CALCULATION
        bennett_free_energy = init_bar(
            energy_fn_template, kbt, 10, 25)

        df, _ = bennett_free_energy(Uvac_traj, Uwat_traj, species, use_subset=use_subset)

        pred_dict[Mol]['predictions'] = df

        counter += 1

elif (load_trajs and check_graph_connection):

    onp.random.seed(7)
    indices_list = onp.arange(len(data))
    onp.random.shuffle(indices_list)
    data = [data[i] for i in indices_list]
    FE_values = [FE_values[i] for i in indices_list]
    smiles = [smiles[i] for i in indices_list]
    GAFF_preds = [GAFF_preds[i] for i in indices_list]
    errors = [errors[i] for i in indices_list]

    data = data[:12]
    FE_values = FE_values[:12]
    smiles = smiles[:12]
    GAFF_preds = GAFF_preds[:12]
    errors = errors[:12]


    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0
    Uvac_connection = []
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

        with open(f"precomputed_trajectories/4300_300ps_load_traj_{Mol}", "rb") as file:
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

        for pos_temp in trajectory.position:
            nbr = nbr.update(pos_temp)
            G = nx.Graph()
            edges_precomputed = [(int(nbr.idx[0][count]), int(nbr.idx[1][count])) for count in
                                 range(onp.sum(nbr.idx[0] != nbr.reference_position.shape[0]))]
            G.add_nodes_from(onp.arange(nbr.reference_position.shape[0]))
            G.add_edges_from(edges_precomputed)
            check_if_one_graph = nx.is_connected(G)
            Uvac_connection.append(check_if_one_graph)

        list_Uvac_connections = [temp for temp in Uvac_connection if temp == False]
        print("Uvac connections if any failed: ", list_Uvac_connections)

        ## WAT
        with open(f"precomputed_trajectories/4300_300ps_load_wat_traj_{Mol}", "rb") as file:
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

        Uwat_connections = []
        for pos_temp in trajectory.position:
            nbr = nbr.update(pos_temp)
            # check graph
            G = nx.Graph()
            edges_precomputed = [(int(nbr.idx[0][count]), int(nbr.idx[1][count])) for count in
                                 range(onp.sum(nbr.idx[0] != nbr.reference_position.shape[0]))]
            G.add_nodes_from(onp.arange(nbr.reference_position.shape[0]))
            G.add_edges_from(edges_precomputed)
            check_if_one_graph = nx.is_connected(G)
            Uwat_connections.append(check_if_one_graph)

        list_Uwat_connections = [temp for temp in Uwat_connections if temp == False]
        print("Uwat connections if any failed: ", list_Uvac_connections)



else: ## LOADING TRAJECTORIES
    # PREDICTION DICTIONARY
    pred_dict = {}
    counter = 0
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
        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uvac_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=init_params, aux=aux)
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
        ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
        Uwat_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                 overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                 barostat_press=barostat_press, entropy_diff=ds,
                                                 free_energy_diff=df, energy_params=loaded_params, aux=aux)

        ## BAR FREE ENERGY CALCULATION
        bennett_free_energy = init_bar(
            energy_fn_template, kbt, 10, 25)

        df, _ = bennett_free_energy(Uvac_traj, Uwat_traj, species)

        pred_dict[Mol]['predictions'] = df

        counter += 1


## SAVING THE PREDICTIONS
with open(save_pred_path, 'wb') as pickle_file:
    pickle.dump(pred_dict, pickle_file)




