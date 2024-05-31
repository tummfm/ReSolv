import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

from jax import value_and_grad, numpy as jnp
import pickle
import numpy as onp
from jax import lax
from jax_md import space
from jax_md import partition


from chemtrain.jax_md_mod import custom_space

save_path = 'ANI1xDB/241122_ANI1x_subset_test.pkl'


# Load trainer and energy function
with open(save_path, 'rb') as pickle_file:
    trainer_loaded = pickle.load(pickle_file)
    energy_fn = trainer_loaded.energy_fn

# Load sample
# Create 100nm^3 box -> Use large box to avoid periodic effects
pad_pos = onp.load('ANI1xDB/pad_pos.npy')[:100]
pad_pos *= 0.1
pad_species = onp.load('ANI1xDB/mask_species.npy')[:100]
box = jnp.eye(3)*100
scale_fn = custom_space.fractional_coordinates_triclinic_box(box)

# Take padded positions. -> Set padded positions such that energy and force contribution is zero.
#  1. Add to non zero values 50nm
#  2. Set padded pos at x=0.6, 1.2, 1.8, .. [nm], y = z = 0. -> Energy contribution is zero
for i, pos in enumerate(pad_pos):
    pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])

for i, pos in enumerate(pad_pos):
    x_spacing = 0
    for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
        x_spacing += 0.6
        pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
print("Done")

# Scale to fractional
pad_pos = lax.map(scale_fn, pad_pos)
pad_species = onp.array(pad_species, dtype='int32')
sample_pos = jnp.array(pad_pos[0])

r_cut = 0.5
displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                      dr_threshold=0.05,
                                      capacity_multiplier=2.,
                                      fractional_coordinates=True,
                                      disable_cell_list=True)

sample_nbrs = neighbor_fn.allocate(sample_pos, extra_capacity=0)
energy, negative_forces = value_and_grad(energy_fn)(sample_pos, neighbor=sample_nbrs, species=jnp.array(pad_species[0]))

print("Energy: ", energy)
print("Forces: ", -negative_forces)
