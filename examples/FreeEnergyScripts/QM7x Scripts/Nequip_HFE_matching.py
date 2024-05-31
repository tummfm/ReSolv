import copy

import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Avoid error in jax 0.4.25
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

from jax import config
config.update("jax_enable_x64", True)
import numpy as onp
import optax
from jax_md import space
from jax import random, numpy as jnp, lax


from util import Initialization
from chemtrain import trainers, util
from chemtrain.jax_md_mod import custom_space

import wandb
import anton_scripts.anton_training_utils as HFE_util
# import TrainFreeEnergy.training_utils_HFE as HFE_util

# GENERAL NOTES: Learn a Nequip potential based onQM7x dataset

if __name__ == '__main__':

    # Choose setup
    model = "NequIP_QM7x_priorInTarget"
    check_freq = None

    # Load data
    # data_indices = HFE_util.get_404_list()
    # data_indices = HFE_util.get_389_train_dataset()
    data_indices = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176, 181, 187, 188, 205, 206,
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


    # exclusion_list = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]

    dataset_for_failed_molecules = [91, 112, 126, 147, 150, 210, 243, 302, 310, 380, 382, 395, 422, 456, 458, 479, 549]
    dataset_for_failed_U_wat = [30, 190, 280, 294, 420]

    exclusion_list = dataset_for_failed_molecules + dataset_for_failed_U_wat

    no_confs = 1
    data, Fe_values, _, _, _ = HFE_util.generate_data(data_indices, no_confs, exclusion_list, "train")
    # data, Fe_values, _, _, _ = HFE_util.generate_data(data_indices, no_confs, exclusion_list, "all")
    pos = [data[i][0] for i in range(len(data))]
    species = [data[i][2] for i in range(len(data))]

    # Pad pos data with zeros
    longest_mol = 0
    max_len = max([len(pos[i]) for i in range(len(pos))])
    for i in range(len(pos)):
        if onp.count_nonzero(pos[i]) > onp.count_nonzero(pos[longest_mol]):
            longest_mol = i
        pos[i] = jnp.pad(pos[i], ((0, max_len - len(pos[i])), (0, 0)))
        species[i] = jnp.pad(species[i], (0, max_len - len(species[i])))
    pad_pos = onp.array(pos)
    pad_species = onp.array(species)
    energies = onp.array(Fe_values)

    # Shuffle the data
    seed = int(sys.argv[6])
    # seed = 10
    indices = onp.arange(len(pad_pos))
    onp.random.shuffle(indices)
    pad_pos = pad_pos[indices]
    pad_species = pad_species[indices]
    energies = energies[indices]

    # Duplicate pad_pos, pad_species, and energies
    pad_pos = onp.concatenate([pad_pos, pad_pos], axis=0)
    pad_species = onp.concatenate([pad_species, pad_species], axis=0)
    energies = onp.concatenate([energies, energies], axis=0)

    # TODO- Reset size based on training dataset.
    # train_ratio = 0.451  # Yields 365 samples for training
    # val_ratio = 0.05     # Yields 39 samples for validation
    train_ratio = 0.501  # Yields 365 samples for training
    val_ratio = 0.05
    batch_size = int(sys.argv[5])
    # date = "110524" -> exlcludes 420
    # date = "120524"  # -> includes 420
    # date = "170524"  # -> excludes 420
    date = "220524"
    initial_lr = float(sys.argv[2])
    lr_decay = float(sys.argv[3])
    num_epochs = int(sys.argv[4])

    wandb.init(
        # set the wandb project where this run will be logged
        project="Direct_HFE_matching",

        # track hyperparameters and run metadata
        config={
            "date": date,
            "initial_lr": str(initial_lr),
            "lr_decay": str(lr_decay),
            "num_epochs": str(num_epochs)
        }
    )

    all_size = pad_pos.shape[0]
    train_size = int(all_size * train_ratio)
    num_transition_steps = int(train_size * num_epochs * (1 / batch_size))

    # Implemenet 5-fold cross-validation
    size_train_validation = int(train_ratio * all_size) + int(val_ratio * all_size)
    fold_number = int(sys.argv[7])
    if fold_number == 1:
        min_num_1 = 0
        max_num_1 = 300
        min_num_2 = 0
        max_num_2 = 0
    elif fold_number == 2:
        min_num_1 = 75
        max_num_1 = 375
        min_num_2 = 0
        max_num_2 = 0
    elif fold_number == 3:
        min_num_1 = 0
        max_num_1 = 75
        min_num_2 = 150
        max_num_2 = 375
    elif fold_number == 4:
        min_num_1 = 0
        max_num_1 = 150
        min_num_2 = 225
        max_num_2 = 375
    elif fold_number == 5:
        min_num_1 = 0
        max_num_1 = 225
        min_num_2 = 300
        max_num_2 = 375
    elif fold_number == 0:
        pass

    if fold_number in (1,2,3,4,5):
        pad_pos_1 = pad_pos[min_num_1:max_num_1]
        pad_species_1 = pad_species[min_num_1:max_num_1]
        energies_1 = energies[min_num_1:max_num_1]
        pad_pos_2 = pad_pos[min_num_2:max_num_2]
        pad_species_2 = pad_species[min_num_2:max_num_2]
        energies_2 = energies[min_num_2:max_num_2]

        pad_pos = onp.concatenate((pad_pos_1, pad_pos_2), axis=0)
        pad_species = onp.concatenate((pad_species_1, pad_species_2), axis = 0)
        energies = onp.concatenate((energies_1, energies_2), axis = 0)

    # Details
    batch_per_device = batch_size
    batch_cache = 10

    id_num = ("Nequip_HFE_matching_date"+str(date)+"_iL" + str(initial_lr) + "_lrd" + str(lr_decay) + "_epochs" +
              str(num_epochs) + '_batch_size' + str(batch_size) + "seed_" + str(seed) + "_fold_number_" + str(fold_number))

    shift_b = "False"
    scale_b = "False"
    mlp = "4"
    train_on = "HFE"
    save_path = "savedTrainers/"+date+"_"+id_num+".pkl"
    save_name = date+"_"+id_num


    # # Adam Belief optimizer
    lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, lr_decay)
    optimizer = optax.chain(
        optax.scale_by_belief(),
        optax.scale_by_schedule(lr_schedule)
    )

    epochs_all = num_epochs

    # for i, pos in enumerate(pad_pos):
    #     pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([500., 500., 500.])
    for i, pos in enumerate(pad_pos):
        x_spacing = 0
        for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
            x_spacing += 15.0
            pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
    box = jnp.eye(3)*1000

    scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
    pad_pos = lax.map(scale_fn, pad_pos)

    # Turn all arrays to jax.numpy format
    energies = jnp.array(energies)
    pad_species = jnp.array(pad_species)
    pad_pos = jnp.array(pad_pos)

    # TODO - Pick R_init, spec_init. Be aware that in select_model() neighborlist needs to be handcrafted on first molecule!!!
    R_init = pad_pos[longest_mol]
    spec_init = pad_species[longest_mol]

    model_init_key = random.PRNGKey(0)
    # key, subkey = random.split(model_init_key)
    #
    # # shuffle positions, energies, HFE exactly the same
    # shuffled_indices = random.shuffle(subkey, onp.arange(0, len(pad_pos)))
    # energies = energies[shuffled_indices]
    # pad_pos = pad_pos[shuffled_indices]
    # pad_species = pad_species[shuffled_indices]

    # Initial values
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)

    energy_fn_template, _, init_params, nbrs_init = Initialization.select_model(model=model, init_pos=R_init,
                                                                                displacement=displacement,
                                                                                box=box, model_init_key=model_init_key,
                                                                                species=spec_init, mol_id_data=None,
                                                                                id_mapping_rev=None)

    trainer = trainers.ForceMatching_QM7x(init_params=init_params, energy_fn_template=energy_fn_template,
                                          nbrs_init=nbrs_init, optimizer=optimizer, position_data=pad_pos,
                                          species_data=pad_species, amber_energy_data=jnp.zeros_like(energies),
                                          amber_force_data=jnp.zeros_like(pad_pos), energy_data=energies, force_data=None,
                                          gamma_f=0, gamma_u=1, batch_per_device=batch_per_device,
                                          batch_cache=batch_cache, train_ratio=train_ratio, val_ratio=val_ratio,
                                          scale_U_F=1, shift_U_F=0)

    # trainer_FM_wrapper = trainers.wrapper_ForceMatching(trainerFM=trainer)
    trainer.train(epochs_all)
    trainer.save_trainer(save_path)

    for count in onp.arange(num_epochs) + 1:
        wandb.log({'train/epochs': count,
                   'train/train_loss': trainer.train_losses[count - 1],
                   'train/val_loss': trainer.val_losses[count - 1]})