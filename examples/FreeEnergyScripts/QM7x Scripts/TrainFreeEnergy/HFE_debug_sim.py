"""Script training GNN potential on experimental Hydration Free Energy data to find a U_water potential function"""
import os
import sys


if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = '-1'

if len(sys.argv) > 2:
    passed_seed = int(str(sys.argv[2]))

os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.97'
#os.environ['JAX_DISABLE_JIT'] = '1'

## TRAINING UTILS NOTABLY CONTAINS A FUNCTION TO CREATE THE DATA SET
## I.E. CONFIGURATIONS, MASSES, SPECIES ETC. USED FOR THE TRAINING
from training_utils_HFE import *

import jax.numpy as jnp
import optax
from jax_md import space
import numpy as onp

from chemtrain import trainers, traj_util
from util import Initialization


## DOUBLE PRECISION
from jax import config
config.update('jax_enable_x64', True)

## NUMBER OF EPOCHS FOR TRAINING
num_epochs = 500
t_prod_U_vac = 250
t_equil_U_vac = 50
print_every_time = 5
date = "080524"
t_prod = 250
t_equil = 50


## EXAMPLE PATHS FOR SAVING PARAMS AND TO LOAD THE VACUUM PARAMS
initial_lr = 0.000001
lr_decay = 0.1

save_checkpoint = (date + "_t_prod_" + str(t_prod_U_vac) + "ps_t_equil_" +
                   str(t_equil_U_vac) + "ps_iL" + str(initial_lr) + "_lrd" +
                   str(lr_decay) + "_epochs" + str(num_epochs)
                   + "_seed" + str(passed_seed) + "_train_389" +
                   "mem_" + str(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']))


# Loading NN potential params
choose_model = 'U_vac'
# choose_model = 'U_wat'
# choose_model = str(sys.argv[5])
if choose_model == 'U_vac':
    #  vacuum model
    vac_model_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                     'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                     'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                     'Params.pkl'
elif choose_model == 'U_wat':
    # water model
    vac_model_path = 'checkpoints/' + save_checkpoint + f'_epoch499.pkl'

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
time_conversion = round(10**3 / 48.8882129,4) # 1 unit --> 1 ps
time_step = 0.001 * time_conversion  # in ps
total_time = t_prod * time_conversion  # in ps - 20
t_equilib = t_equil * time_conversion  # in ps - 5
print_every = print_every_time * time_conversion  # in ps - 0.1s


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


train_dataset_350 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176, 181,
                     187, 188, 205, 206, 211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 1, 3, 6,
                     13, 18, 23, 49, 63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233,
                     248, 259, 260, 272, 277, 280, 290, 310, 311, 318, 337, 22, 112, 173, 195, 210, 285, 302, 347, 372,
                     382, 20, 123, 184, 234, 268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50,
                     51, 52, 57, 59, 60, 61, 65, 66, 67, 70, 72, 76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107,
                     109, 110, 116, 117, 120, 121, 127, 128, 131, 133, 135, 136, 137, 140, 141, 146, 149, 156, 157, 161,
                     165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203, 208, 209, 213, 214, 218, 221, 228, 235,
                     236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273, 274, 275, 276, 279,
                     284, 287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343,
                     348, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101, 103, 108, 111, 115, 119, 129, 132, 134, 155, 159,
                     170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288, 295, 300, 321, 325, 330, 342,
                     0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293, 29, 186, 323, 30, 166, 190, 198,
                     207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96, 113, 118, 124, 125,
                     138, 142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230,
                     231, 232, 239, 244, 246, 253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335,
                     344, 345, 346, 351, 380, 397, 404, 465, 513]

# dataset_for_failed_molecules = [382, 196, 549, 294, 106, 395, 112, 210, 243, 147, 310, 91, 380, 126, 479]

# dataset_for_failed_molecules = [91, 106, 112, 126, 147, 150, 196, 210, 243, 302, 310, 380,
#                                 382, 395, 422, 456, 458, 479, 549]

rerun_after_50ps_worked = [106, 147, 196, 243, 380]

dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]

# remove_from_training = [91, 106, 112, 126, 196, 210, 243, 310, 380, 382, 395, 479, 549]

exclusion_list = None
no_confs = 1

data, FE_values, smiles, calc_h, errors = generate_data(rerun_after_50ps_worked, no_confs, exclusion_list, data_type='train')

data = data[int(sys.argv[3]):int(sys.argv[4])]
FE_values = FE_values[int(sys.argv[3]):int(sys.argv[4])]
smiles = smiles[int(sys.argv[3]):int(sys.argv[4])]
calc_h = calc_h[int(sys.argv[3]):int(sys.argv[4])]
errors = errors[int(sys.argv[3]):int(sys.argv[4])]


# Shuffle the lists
if passed_seed != 0:
    onp.random.seed(passed_seed)
    indices_list = onp.arange(len(data))
    onp.random.shuffle(indices_list)
    data = [data[i] for i in indices_list]
    FE_values = [FE_values[i] for i in indices_list]
    smiles = [smiles[i] for i in indices_list]
    calc_h = [calc_h[i] for i in indices_list]
    errors = [errors[i] for i in indices_list]
else:
    print("Do not shuffle data.")

box = jnp.eye(3)*1000

train_size = len(data)
print("Len train_size: ", train_size)
batch_size = 1
num_transition_steps = int(train_size * num_epochs * (1 / batch_size))

## LEARNING RATE - EMPIRICALLY FOUND TO PERTURB THE UVAC S.T. THE FREE ENERGY CHANGE IS OF UNIT ORDER
## THIS IS A GOOD STARTING POINT SINCE EXPERIMENTALLY THE FREE ENERGY CHANGES ARE OF UNIT ORDER
lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, lr_decay)
optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule)
    )

## INITIALISING THE TIMINGS
timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)


## ADDING STATEPOINTS PER MOLECULE

counter = 0
## VARIABLE TO PASS TO 'Initialization_HFE.initialize_simulation' TO LOAD TRAJECTORIES
load_trajectories = False  #if True, load trajectories and set initialise_traj to False in add_statepoint_HFE
if load_trajectories:
    initialize_traj = False
else:
    initialize_traj = True

for r_init, mass, species, target_dict, smile, Mol in data:

    print('SMILE: ', smile)
    print('Mol: ', Mol)
    print('Experimental Free Energy:', FE_values[counter])

    displacement, shift_fn = space.periodic_general(
        box, fractional_coordinates=True)


    ## INITIALISING THE SIMULATION DATA - BE SURE TO PASS THE SPECIES - NO MODIFIED ARGUMENTS HERE
    simulation_data = Initialization.InitializationClass(r_init=r_init, box=box, kbt=kbt, masses=mass, dt=time_step,
                                                         species=species)

    print(f'Initialising Simulation {counter}')

    ## FURTHER INITIALISATION OF THE SIMULATION
    ## ADITIONAL VARAIBLES HERE ARE:
    ## vac_model_path TO LOAD THE UVAC MODEL
    ## AND load_trajectories TO INITIALISE THE sim_states ACCORDINGLY FOR LOADED TRAJECTORIES
    reference_state, init_params, simulation_fns, compute_fns, targets = \
                            Initialization.initialize_simulation(simulation_data, model, target_dict,
                                                                 integrator=integrator, kbt_dependent=kbt_dependent,
                                                                 dropout_init_seed=dropout_init_seed,
                                                                 vac_model_path=vac_model_path,
                                                                 load_trajectories=load_trajectories)

    simulator_template, energy_fn_template, neighbor_fn = simulation_fns

    ## INITIALISING THE TRAINER
    ## WE JUST USE ONE TRAINER, SO ONLY INITIALISE ON THE FIRST ITERATION
    if counter == 0:
        ## THE TRAINING HERE IS UNCHANGED FROM THE ORIGINAL CODE
        ## NOTE: sim_batch_size=1 MEANS DURING TRAINING, ONE UPDATE CORRESPONDS TO ONE MOLECULE
        ## sim_batch_size=-1 MEANS TO THE BATCH SIZE IS THE TOTAL NUMER OF STATEPOINTS PER UPDATE
        ## NOTE: trainers_HFE.py USED TO BE CLEARER WHAT FUNCTIONS TO TAKE, BUT OTHERWISE THE DIFFTRE CLASS IS UNCHANGED
        trainer = trainers.Difftre_HFE(init_params,
                           optimizer,
                           reweight_ratio=0.9,
                           sim_batch_size=1,
                           energy_fn_template=energy_fn_template)

    print(f'Adding Statepoint {counter}')

    ## ADDING THE STATEPOINT
    ## initilize_traj SHOULD BE FALSE IF WE ARE LOADING TRAJECTORIES
    ## THE MOL ARGUMENT IS USED TO LOAD THE TRAJECTORIES
    trainer.add_statepoint(energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, compute_fns, reference_state, species,
                       targets=targets, Mol=Mol, initialize_traj=initialize_traj, model_type=choose_model)

    print('Experimental Free Energy:', FE_values[counter])
    counter += 1

if load_trajectories:
    # INITIATE TRAINING
    print('Preparing Training')

    save_epochs = [100, 200, 300, 400, 450, 470, 490, 499]

    wrapped_trainer = trainers.wrapper_trainer_HFE(trainer)
    wrapped_trainer.train(num_epochs, save_checkpoint, save_epochs)