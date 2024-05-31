"""Evaluation of uncertainty predictions for molecular properties."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = str('1,2')
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

from pathlib import Path
import pickle

from jax import numpy as jnp, random
import matplotlib.pyplot as plt

from chemtrain import util, probabilistic, property_prediction

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/property_prediction').mkdir(parents=True, exist_ok=True)

trainer_str = 'data_80_01_01_01.pkl'
saved_path = 'output/property_prediction/' + trainer_str

n_dropout_samples = 8
batch_cache = 4
batch_size = 4

trainer = util.load_trainer(saved_path)
test_data = trainer.test_loader._reference_data
train_data = trainer.train_loader._reference_data

recompute = True
cache_path = f'output/property_prediction/{trainer_str}_post_cache.pkl'
if recompute:
    test_uncertainties, test_predictions = probabilistic.dropout_uq_predictions(
        trainer.batched_model, trainer.best_params, trainer.test_loader,
        n_dropout_samples=n_dropout_samples, batch_size=batch_size,
        batch_cache=batch_cache)

    if trainer.val_loader is not None:
        val_data = trainer.val_loader._reference_data
        val_uncertainties, val_predictions = probabilistic.dropout_uq_predictions(
            trainer.batched_model, trainer.best_params, trainer.val_loader,
            n_dropout_samples=n_dropout_samples, batch_size=batch_size,
            batch_cache=batch_cache)

        true_val_charges = val_data['charges']
        val_mask = val_data['species_mask']
        alpha = probabilistic.uq_calibration(val_uncertainties, true_val_charges,
                                             val_mask)
    else:
        alpha = 1

    with open(cache_path, 'wb') as pickle_file:
        pickle.dump((test_uncertainties, test_predictions, alpha), pickle_file)

else:
    with open(cache_path, 'rb') as pickle_file:
        test_uncertainties, test_predictions, alpha = pickle.load(pickle_file)

train_species = train_data['species']
train_mask = train_data['species_mask']
real_train_species = train_species[train_mask]
species_distribution = jnp.bincount(real_train_species)
proton_nbr = jnp.arange(species_distribution.size)

true_charges = test_data['charges']
mask = test_data['species_mask']
test_species = test_data['species']
real_test_species = test_species[mask]

uq_predictions = jnp.mean(test_uncertainties, axis=1)
uq_std = jnp.std(test_uncertainties, axis=1)
calibrated_uq_std = uq_std * alpha

# per box quantities
particles_per_box = jnp.sum(mask, axis=1)
abs_errors = jnp.abs(uq_predictions - true_charges)
mean_box_error = jnp.sum(abs_errors, axis=1) / particles_per_box
mean_box_std = jnp.sum(uq_std, axis=1) / particles_per_box
mean_box_std_calibrated = jnp.sum(calibrated_uq_std, axis=1) / particles_per_box

# per-box mean species error
unique_test_species = jnp.unique(real_test_species)
mean_per_box_species_errors = property_prediction.per_species_box_errors(
    test_data, abs_errors)
mean_per_box_species_std = property_prediction.per_species_box_errors(
    test_data, calibrated_uq_std)
print('Box-averaged mean per-species error:',
      jnp.mean(mean_per_box_species_errors))

# per atom quantities
uq_predictions = uq_predictions[mask]
uq_std = uq_std[mask]
calibrated_uq_std = calibrated_uq_std[mask]
test_predictions = test_predictions[mask]
true_charges = true_charges[mask]

abs_error = jnp.abs(test_predictions - true_charges)
mae = jnp.mean(abs_error)
print('Test set mae:', mae)

abs_error_uq = jnp.abs(uq_predictions - true_charges)
uq_mae = jnp.mean(abs_error_uq)
print('UQ test set mae:', uq_mae)

# per species quantities
train_frequencies = species_distribution[unique_test_species]
per_species_errors = property_prediction.per_species_results(
    real_test_species, abs_error_uq, unique_test_species)
print('Mean per-species error:', jnp.mean(per_species_errors))

per_species_uncertainty = property_prediction.per_species_results(
    real_test_species, calibrated_uq_std, unique_test_species)

plt.figure()
plt.bar(proton_nbr, species_distribution, log=True)
plt.xlabel('Order number')
plt.ylabel('Trainset occurance')
plt.savefig('output/figures/property_prediction_species_train_distribution_log.png')

plt.figure()
plt.bar(proton_nbr, species_distribution)
plt.xlabel('Order number')
plt.ylabel('Trainset occurance')
plt.savefig('output/figures/property_prediction_species_train_distribution.png')

plt.figure()
plt.scatter(true_charges, uq_predictions)
plt.plot([-1.5, 2.5], [-1.5, 2.5], color='k')
plt.xlabel('True charge')
plt.ylabel('Predicted charge')
plt.savefig('output/figures/property_prediction_true_predicted_scatter.png')

plt.figure()
plt.scatter(abs_error_uq, uq_std)
plt.plot([0., 0.3], [0., 0.3], color='k')
plt.xlabel('True error')
plt.ylabel('Predicted std')
plt.savefig('output/figures/property_prediction_atom_errors.png')

plt.figure()
plt.scatter(abs_error_uq, calibrated_uq_std)
plt.plot([0., 0.3], [0., 0.3], color='k')
plt.xlabel('True error')
plt.ylabel('Calibrated std')
plt.savefig('output/figures/property_prediction_atom_errors_calibrated.png')

print('Correlation coefficient per-atom basis:',
      jnp.corrcoef(abs_error_uq, calibrated_uq_std)[0, 1])

print('Correlation coefficient box mean per-atom error:',
      jnp.corrcoef(mean_box_error, mean_box_std)[0, 1])

print('Correlation coefficient box per-species mean error:',
      jnp.corrcoef(mean_per_box_species_errors, mean_per_box_species_std)[0, 1])

random_drawn_std = random.uniform(random.PRNGKey(0), mean_box_error.shape,
                                  minval=0.02, maxval=0.1)
print('Correlation coefficient random draw box mean per-atom error:',
      jnp.corrcoef(mean_box_error, random_drawn_std)[0, 1])

plt.figure()
plt.scatter(mean_box_error, mean_box_std)
plt.plot([0., 0.3], [0., 0.3], color='k')
plt.xlabel('True error')
plt.ylabel('Predicted std')
plt.savefig('output/figures/property_prediction_box_errors.png')

plt.figure()
plt.scatter(mean_per_box_species_errors, mean_per_box_species_std)
plt.plot([0., 0.3], [0., 0.3], color='k')
plt.xlabel('True error')
plt.ylabel('Predicted std')
plt.savefig('output/figures/property_prediction_box_mean_per_species_errors.png')

plt.figure()
plt.scatter(mean_box_error, mean_box_std_calibrated)
plt.plot([0., 0.3], [0., 0.3], color='k')
plt.xlabel('True error')
plt.ylabel('Calibrated std')
plt.savefig('output/figures/property_prediction_box_errors_calibrated.png')

fig, ax = plt.subplots()
ax.bar(unique_test_species, per_species_errors)
ax.bar(unique_test_species, per_species_uncertainty, alpha=0.5)
# ax2 = ax.twinx()
# ax2.bar(unique_species, train_frequencies)
ax.set_xlabel('Proton number')
ax.set_ylabel('Error / Calibrated STD')
# ax2.set_ylabel('Training set frequency')
plt.savefig('output/figures/property_prediction_per_species_bar_plot.png')
