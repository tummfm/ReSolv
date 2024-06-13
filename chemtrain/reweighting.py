"""Implementation of the reweighting formalism.

Allows re-using existing trajectories.
p(S) = ...
"""
from abc import abstractmethod
import time
import warnings

from coax.utils._jit import jit
from jax import (checkpoint, lax, random, grad, tree_util, vmap,
                 numpy as jnp)
from jax_md import util as jax_md_util

from chemtrain import util, traj_util, traj_quantity, max_likelihood
from chemtrain.jax_md_mod import custom_quantity

import jax_md
import jax_sgmc
import jax
import dill
import numpy as onp

def checkpoint_quantities(compute_fns):
    """Applies checkpoint to all compute_fns to save memory on backward pass."""
    for quantity_key in compute_fns:
        compute_fns[quantity_key] = checkpoint(compute_fns[quantity_key])


def _estimate_effective_samples(weights):
    """Returns the effective sample size after reweighting to
    judge reweighting quality.
    """
    # mask to avoid NaN from log(0) if a few weights are 0.
    weights = jnp.where(weights > 1.e-10, weights, 1.e-10)
    exponent = -jnp.sum(weights * jnp.log(weights))
    return jnp.exp(exponent)


def _build_weights(exponents):
    """Returns weights and the effective sample size from exponents
    of the reweighting formulas in a numerically stable way.
    """

    # The reweighting scheme is a softmax, where the exponent above
    # represents the logits. To improve numerical stability and
    # guard against overflow it is good practice to subtract the
    # max of the exponent using the identity softmax(x + c) =
    # softmax(x). With all values in the exponent <=0, this
    # rules out overflow and the 0 value guarantees a denominator >=1.
    exponents -= jnp.max(exponents)
    prob_ratios = jnp.exp(exponents)
    weights = prob_ratios / jax_md_util.high_precision_sum(prob_ratios)
    n_eff = _estimate_effective_samples(weights)
    return weights, n_eff


## ADAPTED WEIGHTS FUNCTION TO COMPUTE LOG_Z_RATIO FOR ZWANZIG FORMULA
def _build_weights_HFE(exponents):
    """Returns weights and the effective sample size from exponents
    of the reweighting formulas in a numerically stable way.
    """

    # The reweighting scheme is a softmax, where the exponent above
    # represents the logits. To improve numerical stability and
    # guard against overflow it is good practice to subtract the
    # max of the exponent using the identity softmax(x + c) =
    # softmax(x). With all values in the exponent <=0, this
    # rules out overflow and the 0 value guarantees a denominator >=1.
    deltaU_max = jnp.max(exponents)
    exponents -= deltaU_max
    prob_ratios = jnp.exp(exponents)
    sum_ratios = jax_md_util.high_precision_sum(prob_ratios)
    log_Z_ratio = jnp.log(sum_ratios) + deltaU_max - jnp.log(exponents.size)
    weights = prob_ratios / sum_ratios
    n_eff = _estimate_effective_samples(weights)
    return weights, n_eff, log_Z_ratio


def reweight_trajectory(traj, **targets):
    """Computes weights to reweight a trajectory from one thermodynamic
    state point to another.

    This function allows re-using an existing trajectory to compute
    observables at slightly perturbed thermodynamic state points. The
    reference trajectory can be generated at a constant state point or
    at different state points, e.g. via non-equlinibrium MD. Both NVT and
    NPT trajectories are supported, however reweighting currently only
    allows reweighting into the same ensemble. For NVT, the trajectory can
    be reweighted to a different temperature. For NPT, can be reweighted to
    different kbT and/or pressure. We assume quantities not included in
    'targets' to be constant over the trajectory, however this is not ensured
    by the code.
    For reference, implemented are cases 1. - 4. in
    https://www.plumed.org/doc-v2.6/user-doc/html/_r_e_w_e_i_g_h_t__t_e_m_p__p_r_e_s_s.html.

    Args:
        traj: Reference trajectory to be reweighted
        targets: Kwargs containing the targets under 'kT' and/or 'pressure'.
                 If a keyword is not provided, the qunatity is assumed to be
                 and remain constant.

    Returns:
        A tuple (weights, n_eff). Weights can be used to compute
        reweighted observables and n_eff judges the expected
        statistical error from reweighting.
    """
    npt_ensemble = util.is_npt_ensemble(traj.sim_state[0])
    if not npt_ensemble:
        assert 'kT' in targets, 'For NVT, a "kT" target needs to be provided.'
    # Note: if temperature / pressure are supposed to remain constant and are
    # hence not provided in the targets, we set them to the respective reference
    # values. Hence, their contribution to reweighting cancels. This should
    # even be at no additional cost under jit as XLA should easily detect the
    # zero contribution. Same applies to combinations in the NPT ensemble.
    target_kbt = targets.get('kT', traj.thermostat_kbt)
    target_beta = 1. / target_kbt
    reference_betas = 1. / traj.thermostat_kbt

    # temperature reweighting
    if 'energy' not in traj.aux:
        raise ValueError('For reweighting, energies need to be provided '
                         'alongside the trajectory. Add energy to auxilary '
                         'outputs in trajectory generator.')
    exponents = -(target_beta - reference_betas) * traj.aux['energy']

    if npt_ensemble:  # correct for P * V
        assert 'kbT' in targets or 'pressure' in targets, ('At least one target'
                                                           ' needs to be given '
                                                           'for reweighting.')
        target_press = targets.get('pressure', traj.barostat_press)
        target_beta_p = target_beta * target_press
        ref_beta_p = reference_betas * traj.barostat_press
        volumes = traj_quantity.volumes(traj)

        # For constant p, reduces to -V * P_ref * (beta_target - beta_ref)
        # For constant T, reduces to -V * beta_ref * (p_target - p_ref)
        exponents -= volumes * (target_beta_p - ref_beta_p)

    return _build_weights(exponents)


def init_pot_reweight_propagation_fns(energy_fn_template, simulator_template,
                                      neighbor_fn, timings, ref_kbt,
                                      ref_press=None, reweight_ratio=0.9,
                                      npt_ensemble=False, energy_batch_size=10):
    """
    Initializes all functions necessary for trajectory reweighting for
    a single state point.

    Initialized functions include a function that computes weights for a
    given trajectory and a function that propagates the trajectory forward
    if the statistical error does not allow a re-use of the trajectory.
    The propagation function also ensures that generated trajectories
    did not encounter any neighbor list overflow.
    """
    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    reweighting_quantities = {'energy': traj_energy_fn}

    if npt_ensemble:
        # pressure currently only used to print pressure of generated trajectory
        # such that user can ensure correct statepoint of reference trajectory
        pressure_fn = custom_quantity.init_pressure(energy_fn_template)
        reweighting_quantities['pressure'] = pressure_fn

    trajectory_generator = traj_util.trajectory_generator_init(
        simulator_template, energy_fn_template, timings, reweighting_quantities)

    beta = 1. / ref_kbt
    checkpoint_quantities(reweighting_quantities)

    def compute_weights(params, traj_state):
        """Computes weights for the reweighting approach."""

        # reweighting properties (U and pressure) under perturbed potential
        reweight_properties = traj_util.quantity_traj(
            traj_state, reweighting_quantities, params, energy_batch_size)

        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -beta * (reweight_properties['energy']
                            - traj_state.aux['energy'])
        return _build_weights(exponent)

    def trajectory_identity_mapping(inputs):
        """Re-uses trajectory if no recomputation needed."""
        traj_state = inputs[1]
        return traj_state

    def recompute_trajectory(inputs):
        """Recomputes the reference trajectory, starting from the last
        state of the previous trajectory to save equilibration time.
        """
        params, traj_state = inputs
        # give kT here as additional input to be handed through to energy_fn
        # for kbt-dependent potentials
        updated_traj = trajectory_generator(params, traj_state.sim_state,
                                            kT=ref_kbt, pressure=ref_press)
        return updated_traj

    @jit
    def propagation_fn(params, traj_state):
        """Checks if a trajectory can be re-used. If not, a new trajectory
        is generated ensuring trajectories are always valid.
        Takes params and the traj_state as input and returns a
        trajectory valid for reweighting as well as an error code
        indicating if the neighborlist buffer overflowed during trajectory
        generation.
        """
        _, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state.aux['energy'].size
        recompute = n_eff < reweight_ratio * n_snapshots
        propagated_state = lax.cond(recompute,
                                    recompute_trajectory,
                                    trajectory_identity_mapping,
                                    (params, traj_state))
        return propagated_state

    def safe_propagate_traj(params, traj_state):
        """Recomputes trajectory and neighbor list until a trajectory without
        overflow was obtained.
        """
        reset_counter = 0
        while traj_state.overflow:
            warnings.warn('Neighborlist buffer overflowed. '
                          'Initializing larger neighborlist.')
            if reset_counter == 3:  # still overflow after multiple resets
                raise RuntimeError('Multiple neighbor list re-computations did '
                                   'not yield a trajectory without overflow. '
                                   'Consider increasing the neighbor list '
                                   'capacity multiplier.')

            # Note: We restart the simulation from the last trajectory, where
            # we know that no overflow has occured. Re-starting from a state
            # that was generated with overflow is dangerous as overflown
            # particles could cause exploding forces once re-considered.
            # TODO is this smart? Last simstate could be very bad state as it
            #  was computed with overflowing neighborlist
            last_state, _ = traj_state.sim_state
            if jnp.any(jnp.isnan(last_state.position)):
                raise RuntimeError('Last state is NaN. Currently there is no '
                                   'recovering from this. Restart from the last'
                                   ' non-overflown state might help, but comes'
                                   ' at the cost that the reference state is '
                                   'likely not representative.')
            if last_state.position.ndim > 2:
                single_enlarged_nbrs = util.neighbor_allocate(
                    neighbor_fn, util.tree_get_single(last_state))
                enlarged_nbrs = vmap(util.neighbor_update, (None, 0))(
                    single_enlarged_nbrs, last_state)
            else:
                enlarged_nbrs = util.neighbor_allocate(neighbor_fn, last_state)
            reset_traj_state = traj_state.replace(
                sim_state=(last_state, enlarged_nbrs))
            traj_state = recompute_trajectory((params, reset_traj_state))
            reset_counter += 1
        return traj_state

    def propagate(params, old_traj_state):
        """Wrapper around jitted propagation function that ensures that
        if neighbor list buffer overflowed, the trajectory is recomputed and
        the neighbor list size is increased until valid trajectory was obtained.
        Due to the recomputation of the neighbor list, this function cannot be
        jit.
        """
        new_traj_state = propagation_fn(params, old_traj_state)
        new_traj_state = safe_propagate_traj(params, new_traj_state)
        return new_traj_state

    def init_first_traj(params, reference_state):
        """Initializes initial trajectory to start optimization from.

        We dump the initial trajectory for equilibration, as initial
        equilibration usually takes much longer than equilibration time
        of each trajectory. If this is still not sufficient, the simulation
        should equilibrate over the course of subsequent updates.
        """
        dump_traj = trajectory_generator(params, reference_state,
                                         kT=ref_kbt, pressure=ref_press)

        t_start = time.time()
        init_traj = trajectory_generator(params, dump_traj.sim_state,
                                         kT=ref_kbt, pressure=ref_press)
        runtime = (time.time() - t_start) / 60.  # in mins
        init_traj = safe_propagate_traj(params, init_traj)
        return init_traj, runtime

    return init_first_traj, compute_weights, propagate


def init_pot_reweight_propagation_fns_HFE(energy_fn_template, simulator_template,
                                      neighbor_fn, timings, ref_kbt,
                                      ref_press=None, reweight_ratio=0.9,
                                      npt_ensemble=False, energy_batch_size=10,
                                      max_iter_bar=25):
    """
    Initializes all functions necessary for trajectory reweighting for
    a single state point.

    Initialized functions include a function that computes weights for a
    given trajectory and a function that propagates the trajectory forward
    if the statistical error does not allow a re-use of the trajectory.
    The propagation function also ensures that generated trajectories
    did not encounter any neighbor list overflow.
    """

    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    reweighting_quantities = {'energy': traj_energy_fn}

    ## INITIALIING THE BAR FUNCTION
    bennett_free_energy = init_bar(
        energy_fn_template, ref_kbt, energy_batch_size, max_iter_bar)

    if npt_ensemble:
        # pressure currently only used to print pressure of generated trajectory
        # such that user can ensure correct statepoint of reference trajectory
        pressure_fn = custom_quantity.init_pressure(energy_fn_template)
        reweighting_quantities['pressure'] = pressure_fn

    ## HERE WE JIT THE TRAJECTORY GENERATOR AS FOUND THIS TO AVOID MEMORY OVERFLOW
    ## OF THE CPUS WHEN LOADING HUNDREDS OF STATEPOINTS
    trajectory_generator = jax.jit(traj_util.trajectory_generator_init_HFE(
        simulator_template, energy_fn_template, timings, reweighting_quantities))
    # trajectory_generator = traj_util.trajectory_generator_init_HFE(
    #     simulator_template, energy_fn_template, timings, reweighting_quantities)

    beta = 1. / ref_kbt
    checkpoint_quantities(reweighting_quantities)

    def compute_weights(params, traj_state, species, entropy_and_free_energy=False):
        """Computes weights for the reweighting approach."""

        # reweighting properties (U and pressure) under perturbed potential
        reweight_properties = traj_util.quantity_traj_HFE(
            traj_state, reweighting_quantities, params, species, energy_batch_size)

        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -beta * (reweight_properties['energy']
                            - traj_state.aux['energy'])

        weights, N_eff, log_Z_ratio = _build_weights_HFE(exponent)

        if not entropy_and_free_energy:
            return weights, N_eff
        else:
            # Compute the free energy difference and entropy difference to the
            # initial potential model
            free_eng_diff = -1 / beta * log_Z_ratio
            eng_diff = jnp.sum(reweight_properties['energy'].T * weights)
            eng_diff -= jnp.mean(traj_state.aux['energy'])

            entropy = beta * (eng_diff - free_eng_diff)

            entropy += traj_state.entropy_diff
            free_eng_diff += traj_state.free_energy_diff

            return weights, N_eff, entropy, free_eng_diff

    def trajectory_identity_mapping(inputs):
        """Re-uses trajectory if no recomputation needed."""
        traj_state = inputs[1]
        return traj_state

    def recompute_trajectory(inputs):
        """Recomputes the reference trajectory, starting from the last
        state of the previous trajectory to save equilibration time.
        """
        params, traj_state, species = inputs
        # give kT here as additional input to be handed through to energy_fn
        # for kbt-dependent potentials
        updated_traj = trajectory_generator(params, traj_state.sim_state, species,
                                            kT=ref_kbt, pressure=ref_press)

        # Apply the BAR procedure to obtain the free energy difference between
        # the old and new trajectory

        dfe, ds = bennett_free_energy(traj_state, updated_traj, species)
        updated_traj = updated_traj.replace(
            entropy_diff=traj_state.entropy_diff + ds,
            free_energy_diff=traj_state.free_energy_diff + dfe
        )
        return updated_traj


    @jit
    def propagation_fn(params, traj_state, species):
        """Checks if a trajectory can be re-used. If not, a new trajectory
        is generated ensuring trajectories are always valid.
        Takes params and the traj_state as input and returns a
        trajectory valid for reweighting as well as an error code
        indicating if the neighborlist buffer overflowed during trajectory
        generation.
        """
        _, n_eff = compute_weights(params, traj_state, species)
        n_snapshots = traj_state.aux['energy'].size

        # print(f"Set reweight ratio to {reweight_ratio}, n_eff bounded by {reweight_ratio * n_snapshots}")
        recompute = jnp.where(n_eff < reweight_ratio * n_snapshots, True, False)
        jax.debug.print("recompute: {}", recompute)

        # jax.debug.print("[Reweighting] {n_eff} < {n_eff_rew} ? -> Trajectory recomputation is {recompute}", n_eff=n_eff, n_eff_rew=reweight_ratio * n_snapshots, recompute=recompute)
        propagated_state = lax.cond(recompute,
                                    recompute_trajectory,
                                    trajectory_identity_mapping,
                                    (params, traj_state, species))
        return propagated_state

    def safe_propagate_traj(params, traj_state,species):
        """Recomputes trajectory and neighbor list until a trajectory without
        overflow was obtained.
        """
        reset_counter = 0
        while traj_state.overflow:
            warnings.warn('Neighborlist buffer overflowed. '
                          'Initializing larger neighborlist.')
            if reset_counter == 3:  # still overflow after multiple resets
                raise RuntimeError('Multiple neighbor list re-computations did '
                                   'not yield a trajectory without overflow. '
                                   'Consider increasing the neighbor list '
                                   'capacity multiplier.')

            # Note: We restart the simulation from the last trajectory, where
            # we know that no overflow has occured. Re-starting from a state
            # that was generated with overflow is dangerous as overflown
            # particles could cause exploding forces once re-considered.
            # TODO is this smart? Last simstate could be very bad state as it
            #  was computed with overflowing neighborlist
            last_state, _ = traj_state.sim_state
            if jnp.any(jnp.isnan(last_state.position)):
                raise RuntimeError('Last state is NaN. Currently there is no '
                                   'recovering from this. Restart from the last'
                                   ' non-overflown state might help, but comes'
                                   ' at the cost that the reference state is '
                                   'likely not representative.')
            if last_state.position.ndim > 2:
                single_enlarged_nbrs = util.neighbor_allocate(
                    neighbor_fn, util.tree_get_single(last_state))
                enlarged_nbrs = vmap(util.neighbor_update, (None, 0))(
                    single_enlarged_nbrs, last_state)
            else:
                enlarged_nbrs = util.neighbor_allocate(neighbor_fn, last_state)
            reset_traj_state = traj_state.replace(
                sim_state=(last_state, enlarged_nbrs))
            traj_state = recompute_trajectory((params, reset_traj_state,species))
            reset_counter += 1
        return traj_state

    def propagate(params, old_traj_state, species):
        """Wrapper around jitted propagation function that ensures that
        if neighbor list buffer overflowed, the trajectory is recomputed and
        the neighbor list size is increased until valid trajectory was obtained.
        Due to the recomputation of the neighbor list, this function cannot be
        jit.
        """
        new_traj_state = propagation_fn(params, old_traj_state,species)
        new_traj_state = safe_propagate_traj(params, new_traj_state,species)
        return new_traj_state

    def init_first_traj(params, reference_state, species, dump_first=True):
        """Initializes initial trajectory to start optimization from.

        We dump the initial trajectory for equilibration, as initial
        equilibration usually takes much longer than equilibration time
        of each trajectory. If this is still not sufficient, the simulation
        should equilibrate over the course of subsequent updates.
        """
        if dump_first:
            dump_traj = trajectory_generator(params, reference_state, species,
                                             kT=ref_kbt, pressure=ref_press)

            t_start = time.time()
            init_traj = trajectory_generator(params, dump_traj.sim_state, species,
                                             kT=ref_kbt, pressure=ref_press)
            runtime = (time.time() - t_start) / 60.  # in mins
        else:
            t_start = time.time()
            init_traj = trajectory_generator(params, reference_state, species,
                                             kT=ref_kbt, pressure=ref_press)
            runtime = (time.time() - t_start) / 60.  # in mins
        init_traj = safe_propagate_traj(params, init_traj, species)

        # Use the initial trajectory as a reference for entropy and free energy
        init_traj = init_traj.replace(
            entropy_diff=0.0,
            free_energy_diff=0.0,
            energy_params=params)

        print(f"Initialized free energy diff to {init_traj.free_energy_diff}")

        return init_traj, runtime


    return init_first_traj, compute_weights, propagate


def init_default_loss_fn(targets):
    """Initializes the default loss function, where MSE errors of
    destinct quantities are added.

    First, observables are computed via the reweighting scheme.
    These observables can be ndarray valued, e.g. vectors for RDF
    / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in
    "quantities[quantity_key]['target']". This per-quantity loss
    is multiplied by gamma in "quantities[quantity_key]['gamma']".
    The final loss is then the sum over all of these weighted
    per-quantity MSE losses. This function allows both observables that
    are simply ensemble averages of instantaneously fluctuating quantities
    and observables that are more complex functions of one or more quantity
    trajectories. The function computing the observable from trajectories of
    instantaneous fluctuating quantities needs to be provided via in
    "quantities[quantity_key]['traj_fn']". For the simple, but common case of
    an average of a single quantity trajectory, 'traj_fn' is given by
    traj_quantity.init_traj_mean_fn.

    Alternatively, a custom loss_fn can be defined. The custom
    loss_fn needs to have the same input-output signuture as the loss_fn
    implemented here.

    Args:
        targets: The target dict with 'gamma', 'target' and 'traj_fn'
        for each observable defined in 'quantities'.

    Returns:
        The loss_fn taking trajectories of fluctuating properties,
        computing ensemble averages via the reweighting scheme and
        outputs the loss and predicted observables.
    """
    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {}
        # multiply weights by N such that averages can be computed with jnp.mean
        # rather than jnp.sum to allow for a unified formulation of
        # functions that compute observables from a trajectory, whether they
        # use weights or not (e.g. in postprocessing)
        weights *= weights.size

        weighted_quant_trajs = {
            quantity_key: (quantity_snapshots.T * weights).T
            for quantity_key, quantity_snapshots in quantity_trajs.items()
        }

        for target_key in targets:
            average = targets[target_key]['traj_fn'](weighted_quant_trajs)
            loss += targets[target_key]['gamma'] * max_likelihood.mse_loss(
                average, targets[target_key]['target'])
            predictions[target_key] = average
        return loss, predictions
    return loss_fn


def init_default_loss_fn_HFE(targets):
    """Initializes the default loss function, where MSE errors of
    destinct quantities are added.

    First, observables are computed via the reweighting scheme.
    These observables can be ndarray valued, e.g. vectors for RDF
    / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in
    "quantities[quantity_key]['target']". This per-quantity loss
    is multiplied by gamma in "quantities[quantity_key]['gamma']".
    The final loss is then the sum over all of these weighted
    per-quantity MSE losses. This function allows both observables that
    are simply ensemble averages of instantaneously fluctuating quantities
    and observables that are more complex functions of one or more quantity
    trajectories. The function computing the observable from trajectories of
    instantaneous fluctuating quantities needs to be provided via in
    "quantities[quantity_key]['traj_fn']". For the simple, but common case of
    an average of a single quantity trajectory, 'traj_fn' is given by
    traj_quantity.init_traj_mean_fn.

    Alternatively, a custom loss_fn can be defined. The custom
    loss_fn needs to have the same input-output signuture as the loss_fn
    implemented here.

    Args:
        targets: The target dict with 'gamma', 'target' and 'traj_fn'
        for each observable defined in 'quantities'.

    Returns:
        The loss_fn taking trajectories of fluctuating properties,
        computing ensemble averages via the reweighting scheme and
        outputs the loss and predicted observables.
    """

    def loss_fn(quantity_trajs, weights, entropy, free_energy):
        loss = 0.
        predictions = {}
        # multiply weights by N such that averages can be computed with jnp.mean
        # rather than jnp.sum to allow for a unified formulation of
        # functions that compute observables from a trajectory, whether they
        # use weights or not (e.g. in postprocessing)

        for target_key in targets:
            ## ADAPTED TO PASS FREE ENERGY
            if target_key == "free_energy_difference":
                average = free_energy
            else:
                average = targets[target_key]['traj_fn'](
                    quantity_trajs, weights=weights)

            loss += targets[target_key]['gamma'] * max_likelihood.mse_loss(
                average, targets[target_key]['target'])
            predictions[target_key] = average


        return loss, predictions
    return loss_fn


def init_rel_entropy_gradient(energy_fn_template, compute_weights, kbt,
                              vmap_batch_size=10):
    """Initializes a function that computes the relative entropy gradient.

    The computation of the gradient is batched to increase computational
    efficiency.

    Args:
        energy_fn_template: Energy function template
        compute_weights: compute_weights function as initialized from
                         init_pot_reweight_propagation_fns.
        kbt: KbT
        vmap_batch_size: Batch size for

    Returns:
        A function rel_entropy_gradient(params, traj_state, reference_batch),
        which returns the relative entropy gradient of 'params' given a
        generated trajectory saved in 'traj_state' and a reference trajectory
        'reference_batch'.
    """
    beta = 1 / kbt

    @jit
    def rel_entropy_gradient(params, traj_state, reference_batch):
        if traj_state.sim_state[0].position.ndim > 2:
            nbrs_init = util.tree_get_single(traj_state.sim_state[1])
        else:
            nbrs_init = traj_state.sim_state[1]

        def energy(params, position):
            energy_fn = energy_fn_template(params)
            # Note: nbrs update requires constant box, i.e. not yet
            # applicable to npt ensemble
            nbrs = nbrs_init.update(position)
            return energy_fn(position, neighbor=nbrs)

        def weighted_gradient(map_input):
            position, weight = map_input
            snapshot_grad = grad(energy)(params, position)  # dudtheta
            weight_gradient = lambda new_grad: weight * new_grad
            weighted_grad_snapshot = tree_util.tree_map(weight_gradient,
                                                        snapshot_grad)
            return weighted_grad_snapshot

        def add_gradient(map_input):
            batch_gradient = vmap(weighted_gradient)(map_input)
            return util.tree_sum(batch_gradient, axis=0)

        weights, _ = compute_weights(params, traj_state)

        # reshape for batched computations
        batch_weights = weights.reshape((-1, vmap_batch_size))
        traj_shape = traj_state.trajectory.position.shape
        batchwise_gen_traj = traj_state.trajectory.position.reshape(
            (-1, vmap_batch_size, traj_shape[-2], traj_shape[-1]))
        ref_shape = reference_batch.shape
        reference_batches = reference_batch.reshape(
            (-1, vmap_batch_size, ref_shape[-2], ref_shape[-1]))

        # no reweighting for reference data: weights = 1 / N
        ref_weights = jnp.ones(reference_batches.shape[:2]) / (ref_shape[0])

        ref_grad = lax.map(add_gradient, (reference_batches, ref_weights))
        mean_ref_grad = util.tree_sum(ref_grad, axis=0)
        gen_traj_grad = lax.map(add_gradient, (batchwise_gen_traj,
                                               batch_weights))
        mean_gen_grad = util.tree_sum(gen_traj_grad, axis=0)

        combine_grads = lambda x, y: beta * (x - y)
        dtheta = tree_util.tree_map(combine_grads, mean_ref_grad, mean_gen_grad)
        return dtheta
    return rel_entropy_gradient


def init_bar(energy_fn_template,
             ref_kbt,
             energy_batch_size=10,
             max_iter_bar=25):
    """Initializes the free energy and entropy computation via the BAR approach.

     The algorithm from [#wyczalkowski2010] uses the BAR method to derive
     the free energy difference between two trajectories and additionally
     derives the entropy difference via the thermodynamic relation
     $TdS = dU - dF$. This implementation relies on the bisection method to
     solve the implicit equation.

     Args:
         old_traj: trajectory generated with the unperturbed potential
         new_traj: trajectory generated with the perturbed potential

    Returns:
        Returns the new_traj with updated free energy and entropy difference.
        These differences are updated by the differences between the old
        and the new trajectory, such that these values resemble the
        differences to the first trajectory that has been generated.

     .. [#wyczalkowski2010] New Estimators for Calculating Solvation Entropy and Enthalpy and Comparative Assessments of Their Accuracy and Precision. Matthew A. Wyczalkowski, Andreas Vitalis, and Rohit V. Pappu in The Journal of Physical Chemistry B 2010 114 (24), 8166-8180, DOI: 10.1021/jp103050u

     """

    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    reweighting_quantities = {'energy': traj_energy_fn}

    # Helper functions to calculate the free energy and entropy difference via
    # the iterative bar approach

    beta = 1. / ref_kbt
    def _vmap_potential_energy_differences(old_traj, new_traj, species):
        """Performs the reweighting procedure vectorized."""

        @jax_sgmc.util.list_vmap
        def _inner(pair):
            vmap_batch_size = max([1, energy_batch_size // 2])
            traj_state, params = pair
            reweighting_properties = traj_util.quantity_traj_HFE(
                traj_state, reweighting_quantities, params, species, vmap_batch_size)
            return reweighting_properties

        # The BAR method requires the energy difference between the potential
        # for both the perturbed and unperturbed trajectories. We thus have to
        # compute the potential energy of the new potential model on the old
        # trajectory and vice versa.
        return _inner(
            (old_traj, new_traj.energy_params),
            (new_traj, old_traj.energy_params))

    def _fr_free_energy(dV_p, dV_0, df):
        """Returns the forward and reverse estimators for the BAR equation."""
        exponent_p = beta * (dV_p - df)
        exponent_0 = beta * (-dV_0 + df)
        gf = 1.0 / (1 + jnp.exp(exponent_p))
        gr = 1.0 / (1 + jnp.exp(exponent_0))
        return gf, gr #gf-gr=0 find df s.t. gf=gr

    def _bar_residual(df, dV_p, dV_0):
        """Squared residual of the implicit BAR equation. """
        gf, gr = _fr_free_energy(dV_p, dV_0, df)
        sum_gf = jax_md.util.high_precision_sum(gf)
        sum_gr = jax_md.util.high_precision_sum(gr)
        return sum_gf - sum_gr

    def _entropy_equation(df, V_0, V_p, rV_p, rV_0):
        """Computes the entropy difference from both trajectories."""
        dV_p = V_p - rV_0
        dV_0 = rV_p - V_0

        gf_p, gr_p = _fr_free_energy(dV_p, dV_p, df)
        gf_0, gr_0 = _fr_free_energy(dV_0, dV_0, df)

        alpha_0 = jnp.mean(gf_0 * V_0) - jnp.mean(gf_0) * jnp.mean(V_0)
        alpha_0 += jnp.mean(gf_0 * gr_0 * dV_0)

        alpha_p = jnp.mean(gr_p * V_p) - jnp.mean(gr_p) * jnp.mean(V_p)
        alpha_p -= jnp.mean(gf_p * gr_p * dV_p)

        du = alpha_0 - alpha_p
        du /= jnp.mean(gf_0 * gr_0) + jnp.mean(gf_p * gr_p) + 1e-30

        return (du - df) / beta

    def _init_bisection(dV_p, dV_0):
        # Helper function to find valid initial points by extending the search
        # interval if necessary
        def _expand_interval(state, _):
            # Expand the interval if both initial points have a residual with equal
            # sign

            df_p, df_0 = state
            loss_p = _bar_residual(df_p, dV_p, dV_0)
            loss_0 = _bar_residual(df_0, dV_p, dV_0)

            # Extend the search interval by a factor of four if solution is
            # not contained in the interval. Ensure that the interval is
            # increased even if the proposals are equal up to a constant
            extend = 1.5 * jnp.abs(df_p - df_0) + 0.5e-4 * jnp.abs(df_p + df_0) + 1e-8
            extend *= jnp.sign(df_p - df_0)
            extend_interval = jnp.where(
                jnp.sign(loss_p) == jnp.sign(loss_0),
                extend, 0.0)
            df_p += extend_interval
            df_0 -= extend_interval

            return (df_p, df_0), None

        # Initialize the guesses
        exponent_p = -beta * dV_p
        exponent_0 = -beta * dV_0

        exp_p = jnp.exp(exponent_p - jnp.max(exponent_p))
        exp_0 = jnp.exp(exponent_0 - jnp.max(exponent_0))

        df_p = jnp.log(jax_md.util.high_precision_sum(exp_p))
        df_0 = jnp.log(jax_md.util.high_precision_sum(exp_0))

        df_p += jnp.max(exponent_p) - jnp.log(exponent_p.size)
        df_0 += jnp.max(exponent_0) - jnp.log(exponent_0.size)

        df_p *= -1 / beta
        df_0 *= -1 / beta

        (df_p, df_0), _ = lax.scan(_expand_interval, (df_p, df_0), onp.arange(10))

        def _bisection_step(state, idx):
            df_a, df_b = state
            df_c = 0.5 * (df_a + df_b)
            loss_a = _bar_residual(df_a, dV_p, dV_0)
            loss_b = _bar_residual(df_b, dV_p, dV_0)
            loss_c = _bar_residual(df_c, dV_p, dV_0)

            # Keep the point A or B that is on the other side of the zero line
            # from point C,
            # i.e. check if the loss for A and C have the same sign.
            # If this is the case, search in [C, B], otherwise in [A, C].
            new_a = jnp.where(jnp.sign(loss_a) == jnp.sign(loss_c), df_c, df_a)
            new_b = jnp.where(jnp.sign(loss_a) == jnp.sign(loss_c), df_b, df_c)

            # Check if c is already the solution (up to the available precision)
            new_a = jnp.where(loss_c == 0.0, df_c, new_a)
            new_b = jnp.where(loss_c == 0.0, df_c, new_b)

            return (new_a, new_b), loss_c

        return (df_p, df_0), _bisection_step

    def bennett_free_energy(old_traj: traj_util, new_traj: traj_util , species, use_subset=None):
        """Compute the free energy and entropy difference.

         The algorithm from [#wyczalkowski2010] uses the BAR method to derive
         the free energy difference between two trajectories and additionally
         derives the entropy difference via the thermodynamic relation
         $TdS = dU - dF$. This implementation relies on the bisection method to
         solve the implicit equation.

         Args:
             old_traj: trajectory generated with the unperturbed potential
             new_traj: trajectory generated with the perturbed potential

        Returns:
            Returns the new_traj with updated free energy and entropy difference.
            These differences are updated by the differences between the old
            and the new trajectory, such that these values resemble the
            differences to the first trajectory that has been generated.

         .. [#wyczalkowski2010] New Estimators for Calculating Solvation Entropy and Enthalpy and Comparative Assessments of Their Accuracy and Precision. Matthew A. Wyczalkowski, Andreas Vitalis, and Rohit V. Pappu in The Journal of Physical Chemistry B 2010 114 (24), 8166-8180, DOI: 10.1021/jp103050u

         """

        # Calculate the differences in potential energy for both trajectories
        rew_0, rew_p = _vmap_potential_energy_differences(old_traj, new_traj, species)



        if use_subset is not None:
            # Get the potential energy from both trajectory
            V_p = new_traj.aux['energy'][:use_subset]
            V_0 = old_traj.aux['energy'][:use_subset]
            rV_p = rew_0['energy'][:use_subset]
            rV_0 = rew_p['energy'][:use_subset]
        else:
            # Get the potential energy from both trajectory
            V_p = new_traj.aux['energy']
            V_0 = old_traj.aux['energy']
            rV_p = rew_0['energy']
            rV_0 = rew_p['energy']

        # dV_p is the energy difference between the perturbed and unperturbed
        # potential based on the perturbed ensemble, dV_0 is the same
        # difference but on the unperturbed ensemble.

        dV_p = V_p - rV_0
        dV_0 = rV_p - V_0

        init_f, update_f = _init_bisection(dV_p, dV_0)
        (dfe, _), _ = lax.scan(update_f, init_f, onp.arange(max_iter_bar))

        ds = _entropy_equation(dfe, V_0, V_p, rV_p, rV_0)

        return dfe, ds

    return bennett_free_energy


class PropagationBase(max_likelihood.MLETrainerTemplate):
    """Trainer base class for shared functionality whenever (multiple)
    simulations are run during training. Can be used as a template to
    build other trainers. Currently used for DiffTRe and relative entropy.

    We only save the latest generated trajectory for each state point.
    While accumulating trajectories would enable more frequent reweighting,
    this effect is likely minor as past trajectories become exponentially
    less useful with changing potential. Additionally, saving long trajectories
    for each statepoint would increase memory requirements over the course of
    the optimization.
    """
    def __init__(self, init_trainer_state, optimizer, checkpoint_path,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.trajectory_states, self.statepoints = {}, {}, {}
        self.n_statepoints = 0
        self.shuffle_key = random.PRNGKey(0)

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbt,
                         set_key=None, energy_batch_size=10,
                         initialize_traj=True, ref_press=None):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""
        # TODO ref pressure only used in print and to have barostat values.
        #  Reevaluate this parameter of barostat values not used in reweighting
        # TODO document ref_press accordingly

        if set_key is not None:
            key = set_key
            if set_key not in self.statepoints.keys():
                self.n_statepoints += 1
        else:
            key = self.n_statepoints
            self.n_statepoints += 1
        self.statepoints[key] = {'kbT': kbt}
        npt_ensemble = util.is_npt_ensemble(reference_state[0])
        if npt_ensemble: self.statepoints[key]['pressure'] = ref_press

        gen_init_traj, compute_weights, propagate = \
            init_pot_reweight_propagation_fns(energy_fn_template,
                                              simulator_template,
                                              neighbor_fn,
                                              timings,
                                              kbt,
                                              ref_press,
                                              self.reweight_ratio,
                                              npt_ensemble,
                                              energy_batch_size)
        if initialize_traj:
            init_traj, runtime = gen_init_traj(self.params, reference_state)
            print(f'Time for trajectory initialization {key}: {runtime} mins')
            self.trajectory_states[key] = init_traj
        else:
            print('Not initializing the initial trajectory is only valid if '
                  'a checkpoint is loaded. In this case, please be use to add '
                  'state points in the same sequence, otherwise loaded '
                  'trajectories will not match its respective simulations.')

        return key, compute_weights, propagate


    ## NEWLY DEFINED FUNCTION
    def _init_statepoint_HFE(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbt, species, Mol,
                         set_key=None, energy_batch_size=10,
                         initialize_traj=True, ref_press=None, model_type=None):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""
        # TODO ref pressure only used in print and to have barostat values.
        #  Reevaluate this parameter of barostat values not used in reweighting
        # TODO document ref_press accordingly

        if set_key is not None:
            key = set_key
            if set_key not in self.statepoints.keys():
                self.n_statepoints += 1
        else:
            key = self.n_statepoints
            self.n_statepoints += 1
        self.statepoints[key] = {'kbT': kbt}

        ## ADAPTED FOR LOADING TRAJECTORIES
        if initialize_traj:
            npt_ensemble = util.is_npt_ensemble(reference_state[0])
        else:
            npt_ensemble = False
        if npt_ensemble: self.statepoints[key]['pressure'] = ref_press

        gen_init_traj, compute_weights, propagate = \
            init_pot_reweight_propagation_fns_HFE(energy_fn_template,
                                              simulator_template,
                                              neighbor_fn,
                                              timings,
                                              kbt,
                                              ref_press,
                                              self.reweight_ratio,
                                              npt_ensemble,
                                              energy_batch_size)
        if initialize_traj:

            init_traj, runtime = gen_init_traj(self.params, reference_state, species, dump_first=True)
            # init_traj, runtime = gen_init_traj(self.params, reference_state, species, dump_first=False)

            print(f'Time for trajectory initialization {key}: {runtime} mins')

            self.trajectory_states[key] = init_traj

            # Save initialized trajetories with init_traj once and later load precomputed trajectories to train
            traj_dict = {}
            aux = init_traj.aux
            barostat_press = init_traj.barostat_press
            over_flow = init_traj.overflow
            sim_state = init_traj.sim_state[0]
            thermostat_kbt = init_traj.thermostat_kbt
            trajectory = init_traj.trajectory
            traj_dict['aux'] = aux
            traj_dict['barostat_press'] = barostat_press
            traj_dict['over_flow'] = over_flow
            traj_dict['sim_state'] = sim_state
            traj_dict['thermostat_kbt'] = thermostat_kbt
            traj_dict['trajectory'] = trajectory
            traj_dict['ds'] = init_traj.entropy_diff
            traj_dict['df'] = init_traj.free_energy_diff

            if model_type == "U_vac":
                with open(f"precomputed_trajectories/250_50ps_load_traj_{Mol}", "wb") as file:
                    dill.dump(traj_dict, file)
            elif model_type == "U_wat":
                with open(f"precomputed_trajectories/250_50ps_load_wat_traj_{Mol}", "wb") as file:
                    dill.dump(traj_dict, file)


        else:
            ## HERE WE LOAD THE TRAJECTORIES E.G:
            if model_type == "U_vac":
                with open(f"precomputed_trajectories/250_50ps_load_traj_{Mol}", "rb") as file:
                    traj_dict = dill.load(file)
            elif model_type == "U_wat":
                with open(f"precomputed_trajectories/250_50ps_load_wat_traj_{Mol}", "rb") as file:
                    traj_dict = dill.load(file)
            aux = traj_dict['aux']
            barostat_press = traj_dict['barostat_press']
            ds = traj_dict['ds']
            df = traj_dict['df']
            over_flow = traj_dict['over_flow']
            sim_state = traj_dict['sim_state']
            thermostat_kbt = traj_dict['thermostat_kbt']
            trajectory = traj_dict['trajectory']
            pos = sim_state.position
            nbr = reference_state
            nbr = nbr.update(pos)
            sim_state = sim_state, nbr
            ## ADAPTED DATACLASS TO TAKE DF,DS AND ENERGY PARAMS AS ARGUMENTS
            init_traj = traj_util.TrajectoryStateBAR(sim_state=sim_state, trajectory=trajectory,
                                                     overflow=over_flow, thermostat_kbt=thermostat_kbt,
                                                     barostat_press=barostat_press, entropy_diff=ds,
                                                     free_energy_diff=df, energy_params=self.params, aux=aux)

            self.trajectory_states[key] = init_traj

        return key, compute_weights, propagate


    @abstractmethod
    def add_statepoint(self, *args, **kwargs):
        """User interface to add additional state point to train model on."""
        raise NotImplementedError()

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        self.state = self.state.replace(params=loaded_params)

    def get_sim_state(self, key):
        return self.trajectory_states[key].sim_state

    def _get_batch(self):
        """Helper function to re-shuffle simulations and split into batches."""
        self.shuffle_key, used_key = random.split(self.shuffle_key, 2)
        shuffled_indices = random.permutation(used_key, self.n_statepoints)
        if self.sim_batch_size == 1:
            batch_list = jnp.split(shuffled_indices, shuffled_indices.size)
        elif self.sim_batch_size == -1:
            batch_list = jnp.split(shuffled_indices, 1)
        else:
            raise NotImplementedError('Only batch_size = 1 or -1 implemented.')

        return (batch.tolist() for batch in batch_list)

    def _print_measured_statepoint(self):
        """Print meausured kbT (and pressure for npt ensemble) for all
        statepoints to ensure the simulation is indeed carried out at the
        prescribed state point.
        """
        for sim_key, traj in self.trajectory_states.items():
            statepoint = self.statepoints[sim_key]
            measured_kbt = jnp.mean(traj.aux['kbT'])
            if 'pressure' in statepoint:  # NPT
                measured_press = jnp.mean(traj.aux['pressure'])
                press_print = (f' press = {measured_press:.2f} ref_press = '
                               f'{statepoint["pressure"]:.2f}')
            else:
                press_print = ''
            print(f'Statepoint {sim_key}: kbT = {measured_kbt:.3f} ref_kbt = '
                  f'{statepoint["kbT"]:.3f}' + press_print)

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        assert self.n_statepoints > 0, ('Add at least 1 state point via '
                                        '"add_statepoint" to start training.')
        super().train(max_epochs, thresh=thresh,
                      checkpoint_freq=checkpoint_freq)

    @abstractmethod
    def _update(self, batch):
        """Implementation of gradient computation, stepping of the optimizer
        and logging of auxiliary results. Takes batch of simulation indices
        as input.
        """
