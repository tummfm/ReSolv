import jax.numpy as jnp
from jax_md import space, partition

box_vector = jnp.ones(3) * 3
box_tensor = jnp.eye(3) * 3

r_cut = 0.1
_positions = jnp.linspace(0.5, 0.7, 20)
positions = jnp.stack([_positions, _positions, _positions], axis=1)

# periodic_general seems to work with box scalar, array and tensor, even though
# documentation mentions only tensor
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)

# works, even though box_tensor is not mentioned in docs
neighbor_fn = partition.neighbor_list(displacement, box_tensor, r_cut,
                                      0.1 * r_cut, fractional_coordinates=True)

# fails due to issue in cell_list
# neighbor_fn = partition.neighbor_list(displacement, box_vector, r_cut,
#                                       0.1 * r_cut, fractional_coordinates=True)

# works
# neighbor_fn = partition.neighbor_list(displacement, box_vector, r_cut,
#                                       0.1 * r_cut, fractional_coordinates=True,
#                                       disable_cell_list=True)

nbrs = neighbor_fn.allocate(positions)
