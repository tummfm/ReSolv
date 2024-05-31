"""Example prediction of partial charges from a dataset of graph
representations.
"""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str('2,3')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

from functools import partial
from pathlib import Path
import pickle

from jax import random, numpy as jnp, device_count
import matplotlib.pyplot as plt
import optax

from chemtrain import (property_prediction, util, max_likelihood, trainers,
                       dropout)

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/property_prediction').mkdir(parents=True, exist_ok=True)

dataset_str = 'calculators/partial_charge_dataset.pkl'
save_path = 'output/property_prediction/saved.pkl'

dropout_seed = 42
# dropout_seed = None

r_cut = 5.  # A
n_species = 100
model_key = random.PRNGKey(0)
batch_cache = 1
batch_per_device = 4

max_epochs = 100
initial_lr = 0.001
decay_end_factor = 1.
train_ratio = 0.80
# val_ratio = train_ratio * (1. / 7.)
val_ratio = 0.

# in total 3378 samples
with open(dataset_str, 'rb') as f:
    data_graph, padded_charges = pickle.load(f)

targets = {'charges': padded_charges}
dropout_mode = {'output': 0.1, 'interaction': 0.1, 'embedding': 0.1}
dataset_size = util.tree_multiplicity(data_graph)

batches_per_train_epoch = int(dataset_size * train_ratio /
                              (batch_per_device * device_count()))

lr_schedule = optax.exponential_decay(
    -initial_lr, batches_per_train_epoch * max_epochs, decay_end_factor)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

sample_graph = util.tree_get_single(data_graph)
init_fn, predictor = property_prediction.partial_charge_prediction(
    r_cut, dropout_mode=dropout_mode)  # embed_size=32

if dropout_seed is None:
    init_params = init_fn(model_key, sample_graph)
else:
    dropout_init_key = random.PRNGKey(dropout_seed)
    init_params = init_fn(model_key, sample_graph, dropout_key=dropout_init_key)
    init_params = dropout.build_dropout_params(init_params, dropout_init_key)


def error_fn(predictions, batch, mask, test_data=False):
    if test_data:
        return max_likelihood.mae_loss(predictions, batch['charges'], mask)
    else:
        return jnp.sqrt(max_likelihood.mse_loss(predictions, batch['charges'],
                                                mask))


test_error_fn = partial(error_fn, test_data=True)

trainer = trainers.PropertyPrediction(
    error_fn, predictor, init_params, optimizer, data_graph, targets,
    batch_per_device, batch_cache, test_error_fn=test_error_fn,
    train_ratio=train_ratio, val_ratio=val_ratio
)

# print(f'Sample prediction: {trainer.predict(sample_graph)}')

# trainer.evaluate_testset_error()
trainer.train(max_epochs)
trainer.evaluate_testset_error()

trainer.save_trainer(save_path)

plt.figure()
plt.semilogy(trainer.train_batch_losses)
plt.ylabel('Training RMSE')
plt.xlabel('Batch')
plt.savefig('output/figures/property_prediction_train_loss.png')

plt.figure()
plt.plot(trainer.train_losses, label='Train loss')
plt.plot(trainer.val_losses, label='Val loss')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('output/figures/property_prediction_epoch_loss.png')
