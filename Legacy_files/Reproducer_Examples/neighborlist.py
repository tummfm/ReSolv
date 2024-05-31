import jax.numpy as jnp
from jax_md import space, partition

# cases:
# r_cut = 0.25, disable_cell_list = False, capacity_multiplier = 1.5:
#     allocates nbrs.idx (20, 20) -> would expect (20, 19) as same particle is
#     not included by default, but works as nbrs.update also allocates (20, 20)
# r_cut = 0.31, disable_cell_list = False, capacity_multiplier = 1.5:
#     Fails, as nbrs.update tries to allocate (20, 19)
# r_cut = 0.31, disable_cell_list = False, capacity_multiplier = 1.:
#     Works, as initially allocates nbrs.idx correctly to (20, 19),
#     but incorrectly sets overflow to True because it tests for
#     capacity >= max_capacity, which is always true for a "full" neighborlist.
#     This overflow issue therefore also arises if not all particles are
#     connected, e.g. r_cut = 0.1.
# disable_cell_list = True, capacity_multiplier = 1.5:
#     Fails for all cases, where the neighbor list is allocated (20, 20)
#     (i.e. r_cut >= 0.12), because nbrs.update tries to allocate (20, 19)
r_cut = 0.12
disable_cell_list = True
capacity_multiplier = 1.5

box = jnp.ones(3)
_positions = jnp.linspace(0.5, 0.7, 20)
positions = jnp.stack([_positions, _positions, _positions], axis=1)
print('Atom shapes:', positions.shape)
displacement, _ = space.periodic(box)

neighbor_fn = partition.neighbor_list(displacement, box, r_cut, 0.1 * r_cut,
                                      capacity_multiplier=capacity_multiplier,
                                      disable_cell_list=disable_cell_list)
nbrs = neighbor_fn.allocate(positions)

print('Neighbor overflow after initial allocation:', nbrs.did_buffer_overflow)
print('Initial neighbor list shape:', nbrs.idx.shape)
new_nbrs = nbrs.update(positions)
print('Neighbor list shape after update:', new_nbrs.idx.shape)
print('Neighbor overflow after update:', nbrs.did_buffer_overflow)
