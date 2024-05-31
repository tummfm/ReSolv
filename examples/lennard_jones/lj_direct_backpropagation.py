import os
import sys
import time

# TODO probably delete at some point

# TODO maybe on CPU better; is it detrimental to short trajectories?
# Does not solve non-determinism issue
# Also happens in JaxMD notebook LJ_neighborlist for long enough trajectory
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs

# config.update("jax_debug_nans", True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)

import numpy as onp
from jax import tree_util, jit, numpy as jnp, random, jvp

from jax_md import util as jax_md_util, space, simulate
from chemtrain.jax_md_mod import io, custom_energy, custom_space, \
    custom_quantity, custom_simulator
from chemtrain.traj_util import process_printouts, trajectory_generator_init, \
    quantity_traj
from chemtrain import reweighting, util, max_likelihood
from util import Postprocessing, Initialization
import optax
from functools import partial
import pickle

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)

type_fn = jax_md_util.f32

# we use reduced LJ units
# TODO: fix reference simulation: RDF is not correct; bug likely in inputs.
#  Look at eva's branch for inspiration, but when possible return to reduce
#  units afterwards
recompute_reference = True
# Boltzmann_constant = 1.
# system_temperature = 1.
# kbT = system_temperature * Boltzmann_constant
kbT = 2.49435321
mass = 18.0154
# density = 0.85
# particles_per_side = 8
# N_particles = particles_per_side ** 3
reference_params = jnp.array([0.3, 1.], dtype=jnp.float32)
init_params = jnp.array([0.2, 0.8], dtype=jnp.float32)
time_step = 2.e-3
total_time = 10.
t_equilib = 2.
# TODO maybe decrease print_every to have a more fine-grained
#  resolution in the plot grad(sigma) over printouts (see below)
print_every = 0.1   # print_every = 0.01 for better resolution
RDF_cut = 1.5
nbins = 250
integrator = 'nve_gradient_stop'
# optimization parameters:
num_updates = 300
initial_lr = -0.003  # step towards negative gradient direction
lr_schedule = optax.exponential_decay(initial_lr, 200, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

timings_struct = process_printouts(time_step, total_time, t_equilib, print_every)

# side_length = (N_particles / density)**(1. / 3.)
# box = jnp.ones(3) * side_length
# print('Box size length:', side_length)
box, R_init, _, _ = io.load_box('data/confs/SPC_FW_3nm.gro')
# R_init = onp.stack([onp.array(r) for r in onp.ndindex(particles_per_side,
#                                                  particles_per_side,
#                                                  particles_per_side)]
#                    ) / particles_per_side  # create initial unit box
# R_init = jnp.array(R_init, jnp.float32)

key = random.PRNGKey(0)
model_init_key, simuation_init_key = random.split(key, 2)

box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
R_init = scale_fn(R_init)

displacement, shift = space.periodic_general(box)

# reduced LJ units: sigma = eps = 1. Cut-off = 2.5
lj_neighbor_energy = partial(custom_energy.customn_lennard_jones_neighbor_list,
                             displacement, box, fractional=True, r_onset=0.8, r_cutoff=0.9)
neighbor_fn, _ = lj_neighbor_energy(sigma=reference_params[0],
                                    epsilon=reference_params[1])
nbrs_init = neighbor_fn(R_init, extra_capacity=0)


def energy_fn_template(energy_params):
    # we only need to re-create energy_fn, neighbor function can be re-used
    energy = lj_neighbor_energy(sigma=energy_params[0],
                                epsilon=energy_params[1],
                                initialize_neighbor_list=False)
    return jit(energy)


if integrator == 'Nose_Hoover':
    simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift,
                                 dt=time_step, kT=kbT, chain_length=3,
                                 chain_steps=1)
elif integrator == 'Langevin':
    # TODO: are results sensitive to simulator?
    #  Langevin adds noise, which was found to be helpful for
    #  exploding gradients in RNNs
    simulator_template = partial(simulate.nvt_langevin, shift=shift,
                                 dt=time_step, kT=kbT, gamma=1.)
elif integrator == 'nve':
    simulator_template = partial(simulate.nve, shift_fn=shift,
                                 dt=time_step)
elif integrator == 'nve_gradient_stop':
    # TODO: make yourself familiar with the gradient stop. Eva in the Bachelor
    #  thesis has already run many experiments with it. If you set gradient
    #  stop to 0, you have the "true" gradient as
    #  This is likely the best place to start with the work. It would be
    #  interesting to see a plot gradient(sigma) over number of printouts
    #  (with gradient_stop=0). We would expect some steady line over the
    #  number of printouts when chaos is not yet bad, but when chaos kicks in,
    #  the gradient will oscillate and even change its direction, at which point
    #  no more learning is possible.
    #  Without gradient stop, this might blow up very quickly. When increasing
    #  gradient_stop, we expect to delay the explosion of the gradient.
    #
    #  TODO: An interesting analysis is also comparing the quality of the
    #   learned model for different numbers of printouts. We expect some
    #   U-shaped curve: For too short trajectories, the statistical error
    #   is too high. For too long trajectories, gradients are destroyed by
    #   chaos and hinder learning. In between will likely be the best place
    #   to learn. How does this curve look for different grad_stop? The curve
    #   will extend further to more snapshots, but does the minimum error
    #   become smaller? If yes, then grad_stop is actually helping. If not,
    #   then grad_stop is actually useless and only seems to be allowing longer
    #   trajectories, but in fact dampens the impact of all states far in the
    #   future and only considers early states, such that the effective
    #   trajectory length is still small.

    #   TODO: Can we compute the effective trajectory length somehow? I would
    #    assume yes, when using the adjoint method. The we can see the impact
    #    of each state on the gradient when backpropagating. The difference of
    #    the gradient before considering a state and afterwards is the impact
    #    of each state. From this, we can compute the effective state size.
    #
    simulator_template = partial(custom_simulator.nve_gradstop,
                                 shift_fn=shift, dt=time_step, stopratio=0.02)
elif integrator == 'nve_adjoint':
    # TODO: make this run and use adjoint equation to analyse the time evolution
    #  of the gradient
    #  goal: compute lyaponov exponent of the time evolution of the gradient
    #        by saving augmented snapshots of the odeint and compute exponent
    #        in postprocessing. If the lyaponov exponent is much shorter than
    #        the simulation length, this would be the proof that gradients
    #        become chaotic and learning is hence not possible any more.
    #
    #  TODO: It would then be interesting to compute the lyaponov exponent
    #        of the gradient for different stop ratios (this still needs to be
    #        implemented into the adjoint_ode solver). We expect the exponent to
    #        increase with increasing stop ratios
    #
    simulator_template = partial(custom_simulator.nve_adjoint)
else:
    raise NotImplementedError('Integrator string not recognized!')

# initialize function to compute rdf
rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = \
    custom_quantity.rdf_discretization(RDF_cut, nbins)
rdf_struct = custom_quantity.RDFParams(
    None, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF)
rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)

if recompute_reference:
    # control the heating of the initial snapshot to the target temperature.
    # The simulation will oscillate around this value in a NVE ensemble
    # t_timings_struct = difftre.process_printouts(time_step, 100., 99.,
    #                                              print_every)
    # t_sim_template = partial(simulate.nvt_langevin, shift=shift,
    #                              dt=time_step, kT=kbT, gamma=1.)
    energy_fn_init = energy_fn_template(reference_params)
    # t_init, _ = t_sim_template(energy_fn_init)
    # t_state = t_init(simuation_init_key, R_init, mass=mass, neighbor=nbrs_init)
    # # store neighbor list together with current simulation state
    # t_sim_state = (t_state, nbrs_init)
    #
    # t_trajectory_generator = difftre.trajectory_generator_init(t_sim_template,
    #                                                            energy_fn_template,
    #                                                            neighbor_fn,
    #                                                            t_timings_struct)
    # t_start = time.time()
    # t_trajectory = t_trajectory_generator(reference_params, t_sim_state)
    # print('Time for initial heat-equilibration:', time.time() - t_start)
    # final_t_state, final_nbrs = t_trajectory[0]

    # use equilibrated state as init to run with real simulator
    reference_timings = process_printouts(time_step, 100., 10., print_every)
    init, _ = simulator_template(energy_fn_init)
    # state = init(simuation_init_key, final_t_state.position, mass=mass,
    #              neighbor=final_nbrs, velocity=final_t_state.velocity)

    state = init(simuation_init_key, R_init, mass=mass,
                 neighbor=nbrs_init, kT=kbT)#  velocity=final_t_state.velocity)


    # sim_state = (state, final_nbrs)
    sim_state = (state, nbrs_init)
    trajectory_generator = trajectory_generator_init(simulator_template,
                                                     energy_fn_template,
                                                     reference_timings)
    t_start = time.time()
    reference_trajectory = trajectory_generator(reference_params, sim_state)
    print('Time for reference trajectory:', time.time() - t_start)
    # start from well-equilibrated state later on
    eq_state = reference_trajectory.sim_state
    final_nbrs = eq_state[1]
    print('Did buffer overflow? ', final_nbrs.did_buffer_overflow)

    quantity_dict = {'rdf': rdf_fn}

    predictions = quantity_traj(reference_trajectory, quantity_dict,
                                reference_params)
    reference_rdf = jnp.mean(predictions['rdf'], axis=0)

    Path('output/simulations/lj').mkdir(parents=True, exist_ok=True)

    plt.Figure()
    plt.plot(rdf_bin_centers, reference_rdf)
    plt.savefig('output/simulations/lj/reference_rdf')

    onp.savetxt('output/simulations/lj/reference_rdf.csv', reference_rdf)
    with open('output/simulations/lj/init_state.pkl', 'wb') as f:
        pickle.dump(eq_state, f)

    init_sim_state = eq_state
    # print('Test determinism:', eq_state[0].position[10])
    sys.exit()

else:  # only load reference
    eq_state = pickle.load(open('data/lj_chaos/init_state', 'rb'))
    eq_state = tree_util.tree_map(type_fn, eq_state)
    reference_rdf = jnp.array(onp.loadtxt('data/lj_chaos/reference_rdf.csv'))

trajectory_generator = trajectory_generator_init(simulator_template,
                                                 energy_fn_template,
                                                 reference_timings)
t_start = time.time()
reference_trajectory = trajectory_generator(reference_params, sim_state)


def init_prediction_and_grad(reference_rdf, only_sigma=True):

    def loss_fn(params, sim_state):
        """Computed rdf and computes corresponding loss."""

        trajectory = trajectory_generator(params, sim_state)
        predictions = quantity_traj(trajectory, quantity_dict, params)
        predicted_rdf = jnp.mean(predictions['rdf'], axis=0)
        loss_val = max_likelihood.mse_loss(predicted_rdf, reference_rdf)
        new_sim_state = trajectory[0]
        return loss_val, new_sim_state  # TODO is it possible to give auxillary variables to jvp function? With grad it is...

    @jit
    def prediction_and_grad(params, sim_state):
        # computes gradient of sigma via forward-autodifferentiation
        loss_val, grad_sigma, new_sim_state = jvp(loss_fn, (params, sim_state), (jnp.array([1., 0.]),))
        if not only_sigma:
            _, grad_eps = jvp(loss_fn, (params, sim_state),
                                   (jnp.array([0., 1.]),))
        else: grad_eps = 0.
        grad = jnp.array([grad_sigma, grad_eps])
        return loss_val, grad, new_sim_state

    return prediction_and_grad


def update(params, opt_state, grad):
    """Update the parameters."""
    # TODO: implement check for overflow of neighborlist, see difftre.init_step_optimizer
    scaled_grad, new_opt_state = optimizer.update(grad,
                                                  opt_state,
                                                  params)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, opt_state

predicion_and_grad = init_prediction_and_grad(reference_rdf)

# initializations
loss_history = []
opt_state = optimizer.init(init_params)
params = init_params
sim_state = (eq_state, nbrs_init)
for step in range(num_updates):
    start_time = time.time()
    loss_val, grad, sim_state = predicion_and_grad(params, sim_state)
    params, opt_state = update(params, opt_state, grad)
    loss_val.block_until_ready()
    step_time = time.time() - start_time

    loss_history.append(loss_val)
    print("Step {} in {:0.2f} sec".format(step, step_time), 'Loss = ', loss_val, '\n')

    print('Loss = ', loss_val)
    print('Sigma:', params[0], 'Epsilon:', params[1])

    if jnp.isnan(loss_val):  # stop learning when optimization diverged
        print('Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup '
              'causing a NaN trajectory.')
        break
