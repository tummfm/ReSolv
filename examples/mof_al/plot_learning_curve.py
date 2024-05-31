from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as onp

Path('output/figures').mkdir(parents=True, exist_ok=True)

n_pool = 2633
al_batch_size = 8

checkpoint_folder = 'output/property_prediction/checkpoints/'
random_errors_path1 = checkpoint_folder + 'errors_rand1.pkl'
random_errors_path2 = checkpoint_folder + 'errors_rand2.pkl'
random_errors_path3 = checkpoint_folder + 'errors_rand3.pkl'
random_errors_path4 = checkpoint_folder + 'errors_random_300.pkl'
# random_errors_path4 = checkpoint_folder + 'errors_rand4.pkl'
# random_errors_path5 = checkpoint_folder + 'errors_rand5.pkl'
al_errors_path = checkpoint_folder + 'errors.pkl'
al_errors_path_inv = checkpoint_folder + 'errors_al_inv.pkl'
foresight_path = checkpoint_folder + 'errors_true_error.pkl'
foresight_path_longer = checkpoint_folder + 'errors_true_long.pkl'
foresight_path_benchmark = checkpoint_folder + 'errors_full_foresight.pkl'


def get_al_train_losses(path_str, al_start_epoch=500):
    trainer_path = checkpoint_folder + path_str
    with open(trainer_path, 'rb') as f:
        trainer = pickle.load(f)
    train_losses = trainer.train_losses
    return train_losses[al_start_epoch:]


def data_ratio(error_list):
    return onp.arange(len(error_list)) * al_batch_size / n_pool


with open(al_errors_path, 'rb') as f:
    al_std_history, al_test_error_history, al_uq_pred_history, al_per_species = pickle.load(f)

with open(foresight_path_longer, 'rb') as f:
    forsight_std_history_longer, foresight_test_error_history_longer, foresight_uq_pred_history_longer = pickle.load(f)

with open(al_errors_path_inv, 'rb') as f:
    al_std_history_inv, al_test_error_history_inv, al_uq_pred_history_inv = pickle.load(f)

with open(random_errors_path1, 'rb') as f:
    rand_std_history1, rand_test_error_history1, rand_uq_pred_history1 = pickle.load(f)

with open(random_errors_path2, 'rb') as f:
    rand_std_history2, rand_test_error_history2, rand_uq_pred_history2 = pickle.load(f)

with open(random_errors_path3, 'rb') as f:
    rand_std_history3, rand_test_error_history3, rand_uq_pred_history3 = pickle.load(f)

with open(random_errors_path4, 'rb') as f:
    rand_std_history4, rand_test_error_history4, rand_uq_pred_history4, rand_per_species4 = pickle.load(f)

with open(foresight_path, 'rb') as f:
    forsight_std_history, foresight_test_error_history, foresight_uq_pred_history = pickle.load(f)

with open(foresight_path_benchmark, 'rb') as f:
    forsight_bench_std_history, foresight_bench_test_error_history, foresight_bench_uq_pred_history, foresight_bench_per_species = pickle.load(f)

plt.figure()
plt.plot(forsight_bench_std_history, label='AL true benchmark')
plt.plot(forsight_std_history_longer, label='AL true error long')
plt.plot(forsight_std_history, label='AL true error')
plt.plot(al_std_history, label='AL')
plt.plot(al_std_history_inv, label='AL inverse')
plt.plot(rand_std_history1, label='Random1', linestyle=':')
plt.plot(rand_std_history2, label='Random2', linestyle=':')
plt.plot(rand_std_history3, label='Random3', linestyle=':')
plt.plot(rand_std_history4, label='Random4', linestyle=':')
plt.ylabel('Predicted mean STD Pooldata')
plt.xlabel('AL Iteration')
plt.legend()
plt.savefig('output/figures/Predicted_STD_history.png')

plt.figure()
plt.plot(foresight_bench_test_error_history, label='AL true benchmark')
plt.plot(foresight_test_error_history_longer, label='AL true error long')
plt.plot(foresight_test_error_history, label='AL true error')
plt.plot(al_test_error_history, label='AL')
plt.plot(al_test_error_history_inv, label='AL inverse')
plt.plot(rand_test_error_history1, label='Random1', linestyle=':')
plt.plot(rand_test_error_history2, label='Random2', linestyle=':')
plt.plot(rand_test_error_history3, label='Random3', linestyle=':')
plt.plot(rand_test_error_history4, label='Random4', linestyle=':')
plt.axhline(y=0.032214705, color='k', linestyle='--', label='Full Data')  # embed32: 0.03246437
plt.ylabel('Test set error without dropout')
plt.xlabel('AL Iteration')
plt.legend()
plt.savefig('output/figures/Test_set_error_history.png')

plt.figure()
plt.plot(data_ratio(foresight_bench_uq_pred_history), foresight_bench_uq_pred_history, label='AL true error')
# plt.plot(foresight_uq_pred_history_longer, label='AL true error long')
# plt.plot(foresight_uq_pred_history, label='AL true error')
plt.plot(data_ratio(al_uq_pred_history), al_uq_pred_history, label='AL')
# plt.plot(al_uq_pred_history_inv, label='AL inverse')
plt.plot(data_ratio(rand_uq_pred_history1), rand_uq_pred_history1, label='Random1', linestyle=':')
plt.plot(data_ratio(rand_uq_pred_history2), rand_uq_pred_history2, label='Random2', linestyle=':')
plt.plot(data_ratio(rand_uq_pred_history3), rand_uq_pred_history3, label='Random3', linestyle=':')
plt.plot(data_ratio(rand_uq_pred_history4), rand_uq_pred_history4, label='Random4', linestyle=':')
plt.axhline(y=0.01483305, color='k', linestyle='--', label='Full Data')  # embed32: 0.021627655
plt.ylabel('MAE prediction error')
plt.xlabel('Data set percentage')
plt.legend()
plt.savefig('output/figures/UQ_prediction_error_history.png')

# Mean per-species error:
plt.figure()
plt.plot(data_ratio(foresight_bench_per_species), foresight_bench_per_species, label='AL true error')
plt.plot(data_ratio(rand_per_species4), rand_per_species4, label='Random')
plt.plot(data_ratio(al_per_species), al_per_species, label='AL')
plt.axhline(y=0.037005644, color='k', linestyle='--', label='Full Data')  # embed32: 0.05625326
plt.ylabel('Mean absolute per-species error')
plt.xlabel('Pool data percentage')
plt.legend()
plt.savefig('output/figures/Mean_per_species_error_history.pdf')

plt.figure()
plt.plot(get_al_train_losses('trainer_full_foresight.pkl')[::2], label='AL true error benchmark')
plt.plot(get_al_train_losses('trainer_true_long.pkl')[::2], label='AL true error more train')
plt.plot(get_al_train_losses('trainer_true_error.pkl'), label='AL true error')
plt.plot(get_al_train_losses('trainer.pkl'), label='AL')
plt.plot(get_al_train_losses('trainer_al_inv.pkl'), label='AL inverse')
plt.plot(get_al_train_losses('trainer_rand1.pkl'), label='Random1', linestyle=':')
plt.plot(get_al_train_losses('trainer_rand2.pkl'), label='Random2', linestyle=':')
plt.plot(get_al_train_losses('trainer_rand3.pkl'), label='Random3', linestyle=':')
plt.plot(get_al_train_losses('trainer_random_300.pkl')[::2], label='Random4', linestyle=':')
plt.ylabel('Train set MSE')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('output/figures/train_loss_history.png')
