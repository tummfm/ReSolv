"""Script to postprocess the free energy predictions with exclusively BAR estimator (quicker than TI)"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''

## DOUBLE PRECISION
from jax import config
config.update('jax_enable_x64', True)

import numpy as onp
import matplotlib.pyplot as plt
import pickle as pkl


#with open("checkpoints/040324_t_prod_250ps_t_equil_50ps_iL1e-06_lrd0.1_loss_epoch400.pkl", 'rb') as f:
#   loss_list = pkl.load(f)

with open("checkpoints/020224_t_prod_25ps_t_equil_5ps_iL1e-05_lrd0.01_loss_epoch400.pkl", 'rb') as f:
    loss_list = pkl.load(f)

loss_train = onp.array(loss_list)
print(loss_train.shape)

# Plot train loss
plt.plot(loss_train[-400:])
plt.ylim([0, 40])
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss')
plt.show()