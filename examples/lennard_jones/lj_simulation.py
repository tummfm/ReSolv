"""Example running LJ simulation for reference data generation."""
import os
import sys
import time

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

from jax import numpy as jnp, random, value_and_grad, vmap
from jax_md import space, simulate
import numpy as onp

from chemtrain import traj_util
from chemtrain.jax_md_mod import custom_space, custom_quantity, custom_energy

# Dataset parameters: kbT and box length
kbt = 1.
side_length = 10.

# we use reduced LJ units
LJ_params = jnp.array([1., 1.], dtype=jnp.float32)
mass = 1.

time_step = 1.e-3
total_time = 1000.1
t_equilib = 100.
print_every = 1.
rdf_cut = 5.
nbins = int(rdf_cut * 100)

n_particles = 1000
particles_per_side = int(round(n_particles ** (1. / 3.)))

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

box = jnp.ones(3) * side_length

R_init = onp.stack([onp.array(r) for r in onp.ndindex(particles_per_side,
                                                      particles_per_side,
                                                      particles_per_side)]
                   ) / particles_per_side  # create initial unit box
R_init = jnp.array(R_init, jnp.float32)

model_init_key, simuation_init_key = random.split(random.PRNGKey(0), 2)
box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
displacement, shift = space.periodic_general(box)

r_cut = 2.5
lj_neighbor_energy = partial(custom_energy.customn_lennard_jones_neighbor_list,
                             displacement, box, fractional=True, r_onset=2.,
                             r_cutoff=r_cut, disable_cell_list=True)
neighbor_fn, _ = lj_neighbor_energy(sigma=LJ_params[0], epsilon=LJ_params[1])
nbrs_init = neighbor_fn.allocate(R_init)


def energy_fn_template(energy_params):
    # we only need to re-create energy_fn, neighbor function can be re-used
    energy = lj_neighbor_energy(sigma=energy_params[0],
                                epsilon=energy_params[1],
                                initialize_neighbor_list=False)
    return energy

energy_fn_init = energy_fn_template(LJ_params)
simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift,
                             dt=time_step, kT=kbt)
init, _ = simulator_template(energy_fn_init)
state = init(simuation_init_key, R_init, mass=mass, neighbor=nbrs_init, kT=kbt)
init_sim_state = (state, nbrs_init)

trajectory_generator = traj_util.trajectory_generator_init(
    simulator_template, energy_fn_template, timings)
t_start = time.time()
reference_trajectory = trajectory_generator(LJ_params, init_sim_state)
print('Time for reference trajectory:', time.time() - t_start)
# start from well-equilibrated state later on
end_state = reference_trajectory.sim_state
final_nbrs = end_state[1]
print('Did buffer overflow? ', final_nbrs.did_buffer_overflow)

# initialize function to compute rdf
rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = \
    custom_quantity.rdf_discretization(rdf_cut, nbins)
rdf_struct = custom_quantity.RDFParams(
    None, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF)
rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)
quantity_dict = {'rdf': rdf_fn}

predictions = traj_util.quantity_traj(reference_trajectory, quantity_dict)
reference_rdf = jnp.mean(predictions['rdf'], axis=0)

simulation_str = f'output/simulations/lj/temp_{kbt}_l_{side_length}/'

Path(simulation_str).mkdir(parents=True, exist_ok=True)

plt.Figure()
plt.plot(rdf_bin_centers, reference_rdf)
plt.savefig(simulation_str + 'reference_rdf.png')

rdf_data = jnp.column_stack((rdf_struct.rdf_bin_centers, reference_rdf))

onp.savetxt(simulation_str + 'reference_rdf.csv', rdf_data)

pair_nbrs = neighbor_fn.allocate(jnp.array([[0., 0., 0.], [0., 0., 0.01]]))


def distance_to_positions(distance):
    r_vect = jnp.array([[0., 0., 0.], [0., 0., distance]])
    return r_vect


@vmap
def pairwise_energy_forces(distance):
    r_vect = distance_to_positions(distance)
    r_vect = scale_fn(r_vect)
    energy, neg_forces = value_and_grad(energy_fn_init)(r_vect, pair_nbrs)
    return energy, -neg_forces


distance_vect_ordered = jnp.linspace(0.95 * LJ_params[0], rdf_cut,
                                     int((r_cut - 0.8*LJ_params[0]) * 100 + 1))
ordered_energies, ordered_forces = pairwise_energy_forces(distance_vect_ordered)
ordered_positions = vmap(distance_to_positions)(distance_vect_ordered)

n_dataset_boxes = 10
trajectory = reference_trajectory.trajectory.position
subsampling = int(trajectory.shape[0] / n_dataset_boxes)
trajectory = trajectory[::subsampling]
print(trajectory.shape)


def upper_triangle(mat):
    m = mat.shape[0]
    r, c = jnp.triu_indices(m, 1)
    return mat[r, c]


pair_dist_fn = space.map_product(displacement)
distance_list = []
for i in range(trajectory.shape[0]):
    positions = trajectory[i]
    pair_distances = space.distance(pair_dist_fn(positions, positions))
    distances = upper_triangle(pair_distances)
    within_cut_dist = distances[(distances < r_cut).nonzero()]
    distance_list.append(within_cut_dist)

distance_vect = jnp.concatenate(distance_list)
dataset_positions = vmap(distance_to_positions)(distance_vect)
dataset_energies, dataset_forces = pairwise_energy_forces(distance_vect)

onp.savez(simulation_str + 'dataset.npz', box=box_tensor,
          positions=dataset_positions, energies=dataset_energies,
          forces=dataset_forces)

onp.savez(simulation_str + 'ordered_points.npz', box=box_tensor,
          positions=ordered_positions, energies=ordered_positions,
          forces=ordered_positions)


plt.figure()
plt.plot(distance_vect_ordered, ordered_energies)
plt.savefig(simulation_str + 'energy_selection.png')

