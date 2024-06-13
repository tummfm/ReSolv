"""This file contains several Trainer classes as a quickstart for users."""
import warnings

from blackjax import nuts, stan_warmup
from coax.utils._jit import jit
from jax import value_and_grad, random, numpy as jnp
# from jax import tree_util
from jax_sgmc import data
from jax_sgmc.data import numpy_loader
import jax
import pickle
import numpy as onp
import subprocess
# import gc
# import functools
import networkx as nx

from chemtrain import (util, force_matching, traj_util, reweighting,
                       probabilistic, max_likelihood, property_prediction)

from chemtrain import jax_md_mod


class PropertyPrediction(max_likelihood.DataParallelTrainer):
    """Trainer for direct prediction of molecular properties."""
    def __init__(self, error_fn, prediction_model, init_params, optimizer,
                 graph_dataset, targets, batch_per_device=1, batch_cache=10,
                 train_ratio=0.7, val_ratio=0.1, test_error_fn=None,
                 shuffle=False, convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        # TODO documentation

        # TODO build graph on-the-fly as memory moving might be bottleneck here
        model = property_prediction.init_model(prediction_model)
        checkpoint_path = 'output/property_prediction/' + str(checkpoint_folder)
        dataset_dict = {'targets': targets, 'graph_dataset': graph_dataset}
        loss_fn = property_prediction.init_loss_fn(error_fn)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion)

        self.test_error_fn = test_error_fn
        self._init_test_fn()

    @staticmethod
    def _build_dataset(targets, graph_dataset):
        return property_prediction.build_dataset(targets, graph_dataset)

    def predict(self, single_observation):
        """Prediction for a single input graph using the current param state."""
        # TODO jit somewhere?
        return self.model(self.best_inference_params, single_observation)

    def evaluate_testset_error(self, best_params=True):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios.')
        assert self._test_fn is not None, ('"test_error_fn" is necessary'
                                           ' during initialization.')

        params = (self.best_inference_params_replicated
                  if best_params else self.state.params)
        error = self._test_fn(params)
        print(f'Error on test set: {error}')
        return error

    def _init_test_fn(self):
        if self.test_error_fn is not None and self.test_loader is not None:
            test_loss_fn = property_prediction.init_loss_fn(self.test_error_fn)
            self._test_fn, data_release_fn = max_likelihood.init_val_loss_fn(
                self.batched_model, test_loss_fn, self.test_loader,
                self.target_keys, self.batch_size, self.batch_cache)
            self.release_fns.append(data_release_fn)
        else:
            self._test_fn = None


class ForceMatching(max_likelihood.DataParallelTrainer):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.
    # TODO generalize to padded particles and without neighborlists

    Virial data is pressure tensor, i.e. negative stress tensor

    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, energy_data=None, force_data=None,
                 virial_data=None, box_tensor=None, gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, shuffle=False,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        dataset_dict = {'position_data': position_data,
                        'energy_data': energy_data,
                        'force_data': force_data,
                        'virial_data': virial_data
                        }

        virial_fn = force_matching.init_virial_fn(
            virial_data, energy_fn_template, box_tensor)
        model = force_matching.init_model(nbrs_init, energy_fn_template,
                                          virial_fn)
        loss_fn = force_matching.init_loss_fn(gamma_f=gamma_f, gamma_p=gamma_p)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion,
                         energy_fn_template=energy_fn_template)
        self._virial_fn = virial_fn
        self._nbrs_init = nbrs_init
        self._init_test_fn()

    @staticmethod
    def _build_dataset(position_data, energy_data=None, force_data=None,
                       virial_data=None):
        return force_matching.build_dataset(position_data, energy_data,
                                            force_data, virial_data)

    def evaluate_mae_testset(self):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios or add a'
                                              ' test_loader manually.')
        maes = self.mae_fn(self.best_inference_params_replicated)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')

    def _init_test_fn(self):
        if self.test_loader is not None:
            self.mae_fn, data_release_fn = force_matching.init_mae_fn(
                self.test_loader, self._nbrs_init,
                self.reference_energy_fn_template, self.batch_size,
                self.batch_cache, self._virial_fn
            )
            self.release_fns.append(data_release_fn)
        else:
            self.mae_fn = None


class ForceMatching_ANI1x(max_likelihood.DataParallelTrainer):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.
    # TODO generalize to padded particles and without neighborlists

    Virial data is pressure tensor, i.e. negative stress tensor

    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, species_data, energy_data=None, force_data=None,
                 virial_data=None, box_tensor=None, gamma_u=1., gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, shuffle=False,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        dataset_dict = {'position_data': position_data,
                        'species_data': species_data,
                        'energy_data': energy_data,
                        'force_data': force_data,
                        'virial_data': virial_data
                        }

        virial_fn = force_matching.init_virial_fn(
            virial_data, energy_fn_template, box_tensor)
        model = force_matching.init_model_ANI1x(nbrs_init, energy_fn_template,
                                                virial_fn)
        # loss_fn = force_matching.init_loss_fn(gamma_u=gamma_u, gamma_f=gamma_f, gamma_p=gamma_p)
        loss_fn = force_matching.init_loss_fn_scaled(gamma_u=gamma_u, gamma_f=gamma_f, gamma_p=gamma_p)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion,
                         energy_fn_template=energy_fn_template)
        self._virial_fn = virial_fn
        self._nbrs_init = nbrs_init
        self._init_test_fn()

    @staticmethod
    def _build_dataset(position_data, species_data, energy_data=None, force_data=None,
                       virial_data=None):
        return force_matching.build_dataset_ANI1x(position_data, species_data, energy_data,
                                                  force_data, virial_data)

    def evaluate_mae_testset(self):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios or add a'
                                              ' test_loader manually.')
        maes = self.mae_fn(self.best_inference_params_replicated)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')

    def _init_test_fn(self):
        if self.test_loader is not None:
            self.mae_fn, data_release_fn = force_matching.init_mae_fn(
                self.test_loader, self._nbrs_init,
                self.reference_energy_fn_template, self.batch_size,
                self.batch_cache, self._virial_fn
            )
            self.release_fns.append(data_release_fn)
        else:
            self.mae_fn = None


class ForceMatching_QM7x(max_likelihood.DataParallelTrainer):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.
    # TODO generalize to padded particles and without neighborlists

    Virial data is pressure tensor, i.e. negative stress tensor

    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, species_data, amber_energy_data, amber_force_data, energy_data=None, force_data=None,
                 virial_data=None, box_tensor=None, gamma_u=1., gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, shuffle=False,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints',
                 scale_U_F=1., shift_U_F=0.):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        dataset_dict = {'position_data': position_data,
                        'species_data': species_data,
                        'energy_data': energy_data,
                        'force_data': force_data,
                        'virial_data': virial_data,
                        'amber_energy_data': amber_energy_data,
                        'amber_force_data': amber_force_data
                        }

        virial_fn = force_matching.init_virial_fn(
            virial_data, energy_fn_template, box_tensor)
        model = force_matching.init_model_QM7x(nbrs_init, energy_fn_template,
                                                virial_fn)
        # loss_fn = force_matching.init_loss_fn(gamma_u=gamma_u, gamma_f=gamma_f, gamma_p=gamma_p)
        loss_fn = force_matching.init_loss_fn_scaled(gamma_u=gamma_u, gamma_f=gamma_f, gamma_p=gamma_p,
                                                     error_fn=max_likelihood.mse_loss,
                                                     scale_U_F=scale_U_F, shift_U_F=shift_U_F)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion,
                         energy_fn_template=energy_fn_template)
        self._virial_fn = virial_fn
        self._nbrs_init = nbrs_init
        self._init_test_fn()

    @staticmethod
    def _build_dataset(position_data, species_data, energy_data=None, force_data=None,
                       virial_data=None, amber_energy_data=None, amber_force_data=None):
        return force_matching.build_dataset_QM7x(position_data, species_data, energy_data,
                                                 force_data, virial_data, amber_energy_data=amber_energy_data,
                                                 amber_force_data=amber_force_data)

    def evaluate_mae_testset(self):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios or add a'
                                              ' test_loader manually.')
        maes = self.mae_fn(self.best_inference_params_replicated)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')

    def _init_test_fn(self):
        if self.test_loader is not None:
            self.mae_fn, data_release_fn = force_matching.init_mae_fn(
                self.test_loader, self._nbrs_init,
                self.reference_energy_fn_template, self.batch_size,
                self.batch_cache, self._virial_fn
            )
            self.release_fns.append(data_release_fn)
        else:
            self.mae_fn = None



class Difftre(reweighting.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method."""
    def __init__(self, init_params, optimizer, reweight_ratio=0.9,
                 sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """Initializes a DiffTRe trainer instance.

        The implementation assumes a NVT ensemble in weight computation.
        The trainer initialization only sets the initial trainer state
        as well as checkpointing and save-functionality. For training,
        target state points with respective simulations need to be added
        via 'add_statepoint'.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes a new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the maximum loss
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average loss across the batch. For a
                                   single state point, both are equivalent.
                                   A criterion based on the rolling standatd
                                   deviation 'std' might be implemented in the
                                   future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        self.batch_losses, self.epoch_losses = [], []
        self.predictions = {}
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size,
            energy_fn_template=energy_fn_template)

        self.early_stop = max_likelihood.EarlyStopping(self.params,
                                                       convergence_criterion)

    def add_statepoint(self, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, quantities,
                       reference_state, targets=None, ref_press=None,
                       loss_fn=None, vmap_batch=10, initialize_traj=True,
                       set_key=None):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. In case the default loss function should
        be employed, for each observable the 'target' dict containing
        a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target' needs to be provided.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness),
        a custom loss_fn needs to be defined. The signature of the loss_fn
        needs to be the following: It takes the trajectory of computed
        instantaneous quantities saved in a dict under its respective key of
        the quantities_dict. Additionally, it receives corresponding weights
        w_i to perform ensemble averages under the reweighting scheme. With
        these components, ensemble averages of more complex observables can
        be computed. The output of the function is (loss value, predicted
        ensemble averages). The latter is only necessary for post-processing
        the optimization process. See 'init_independent_mse_loss_fn' for
        an example implementation.

        Args:
            energy_fn_template: Function that takes energy parameters and
                                initializes a new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            quantities: Dict containing for each observable specified by the
                        key a corresponding function to compute it for each
                        snapshot using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                     another dict providing 'gamma' and 'target' for each
                     observable. Targets are only necessary when using the
                     'independent_loss_fn'.
            loss_fn: Custom loss function taking the trajectory of quantities
                     and weights and returning the loss and predictions;
                     Default None initializes an independent MSE loss, which
                     computes reweighting averages from snapshot-based
                     observables.
            vmap_batch: Batch size of vmapping of per-snapshot energy for weight
                        computation.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                     By default, uses the index of the sequance statepoints are
                     added, i.e. self.trajectory_states[0] for the first added
                     statepoint. Can be used for changing the timings of the
                     simulation during training.
        """

        # init simulation, reweighting functions and initial trajectory
        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           set_key,
                                                           vmap_batch,
                                                           initialize_traj,
                                                           ref_press)

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = reweighting.init_default_loss_fn(targets)
        else:
            print('Using custom loss function. Ignoring "target" dict.')

        reweighting.checkpoint_quantities(quantities)

        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            weights, _ = weights_fn(params, traj_state)
            quantity_trajs = traj_util.quantity_traj(traj_state,
                                                     quantities,
                                                     params)
            loss, predictions = loss_fn(quantity_trajs, weights)
            return loss, predictions

        statepoint_grad_fn = jit(value_and_grad(difftre_loss, has_aux=True))

        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state)
            outputs, grad = statepoint_grad_fn(params, traj_state)
            loss_val, predictions = outputs
            return traj_state, grad, loss_val, predictions

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts
        grads, losses = [], []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]
            new_traj_state, curr_grad, loss_val, state_point_predictions = \
                grad_fn(self.params, self.trajectory_states[sim_key])

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = state_point_predictions
            grads.append(curr_grad)
            losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self._diverged = True  # ends training
                break

        self.batch_losses.append(sum(losses) / self.sim_batch_size)
        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)
        self.gradient_norm_history.append(util.tree_norm(batch_grad))

    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(f'\nEpoch {self._epoch}: Epoch loss = {epoch_loss:.5f}, Gradient '
              f'norm: {self.gradient_norm_history[-1]}, '
              f'Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        # print last scalar predictions
        for statepoint, prediction_series in self.predictions.items():
            last_predictions = prediction_series[self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    print(f'Statepoint {statepoint}: Predicted {quantity}:'
                          f' {value}')

        self._converged = self.early_stop.early_stopping(epoch_loss, thresh,
                                                         self.params)

    @property
    def best_params(self):
        return self.early_stop.best_params

    def move_to_device(self):
        super().move_to_device()
        self.early_stop.move_to_device()


class Difftre_HFE(reweighting.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method."""
    def __init__(self, init_params, optimizer, reweight_ratio=0.9,
                 sim_batch_size=1, energy_fn_template=None, loaded_params=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """Initializes a DiffTRe trainer instance.

        The implementation assumes a NVT ensemble in weight computation.
        The trainer initialization only sets the initial trainer state
        as well as checkpointing and save-functionality. For training,
        target state points with respective simulations need to be added
        via 'add_statepoint'.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes a new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the maximum loss
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average loss across the batch. For a
                                   single state point, both are equivalent.
                                   A criterion based on the rolling standatd
                                   deviation 'std' might be implemented in the
                                   future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        self.batch_losses, self.epoch_losses = [], []
        self.predictions = {}
        self._targets = {}
        self.cache_size = []
        self.update_counter = 0
        # self.gpu_memory_list = []
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size,
            energy_fn_template=energy_fn_template)

        self.early_stop = max_likelihood.EarlyStopping(self.params,
                                                       convergence_criterion)


    ## NEWLY DEFINED FUNCTION
    ## ADDITIONALLY PASS 'Mol' WHICH IS THE MOLECULE KEY TO LOAD THE PRECOMPUTED TRAJECTORY
    def add_statepoint(self, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, quantities,
                       reference_state, species=None, targets=None, Mol=None, ref_press=None,
                       loss_fn=None, vmap_batch=10, initialize_traj=True,
                       set_key=None, loss_kwars=None, model_type=None):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. In case the default loss function should
        be employed, for each observable the 'target' dict containing
        a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target' needs to be provided.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness),
        a custom loss_fn needs to be defined. The signature of the loss_fn
        needs to be the following: It takes the trajectory of computed
        instantaneous quantities saved in a dict under its respective key of
        the quantities_dict. Additionally, it receives corresponding weights
        w_i to perform ensemble averages under the reweighting scheme. With
        these components, ensemble averages of more complex observables can
        be computed. The output of the function is (loss value, predicted
        ensemble averages). The latter is only necessary for post-processing
        the optimization process. See 'init_independent_mse_loss_fn' for
        an example implementation.

        Args:
            energy_fn_template: Function that takes energy parameters and
                                initializes a new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            quantities: Dict containing for each observable specified by the
                        key a corresponding function to compute it for each
                        snapshot using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                     another dict providing 'gamma' and 'target' for each
                     observable. Targets are only necessary when using the
                     'independent_loss_fn'.
            loss_fn: Custom loss function taking the trajectory of quantities
                     and weights and returning the loss and predictions;
                     Default None initializes an independent MSE loss, which
                     computes reweighting averages from snapshot-based
                     observables.
            vmap_batch: Batch size of vmapping of per-snapshot energy for weight
                        computation.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                     By default, uses the index of the sequance statepoints are
                     added, i.e. self.trajectory_states[0] for the first added
                     statepoint. Can be used for changing the timings of the
                     simulation during training.
        """
        if loss_kwars is None:
            loss_kwars = {}

        # init simulation, reweighting functions and initial trajectory
        ## COMPARED TO ORIGINAL CODE, ADDITIONALLY PASS SPECIES AND MOL KEY
        ## THIS FUNCTION IS FOUND IN 'reweighting_HFE.py'
        key, weights_fn, propagate = self._init_statepoint_HFE(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt, species, Mol,
                                                           set_key,
                                                           vmap_batch,
                                                           initialize_traj,
                                                           ref_press,
                                                           model_type)

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = reweighting.init_default_loss_fn_HFE(targets, **loss_kwars)
        else:
            print('Using custom loss function. Ignoring "target" dict.')


        quantities['energy'] = jax_md_mod.custom_quantity.energy_wrapper(energy_fn_template)

        reweighting.checkpoint_quantities(quantities)

        ## DIFFTRE LOSS FUNCTION ADAPTED TO RETURN ENTROPY AND FREE ENERGY
        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            #WEIGHTS_FN ADDITIONALLY RETURNS  ENTROPY AND FREE ENERGY
            weights, _, entropy, free_energy = weights_fn(
                params, traj_state, species, entropy_and_free_energy=True)

            ## THE QUANTITY TRAJ FN IS THE SAME, BUT ADAPTED TO TAKE SPECIES
            ## FOUND IN traj_util.py, line 938
            quantity_trajs = traj_util.quantity_traj_HFE(traj_state,
                                                     quantities,
                                                     params, species)
            ## THE LOSS FN IS THE SAME, BUT ADAPTED TO TAKE ENTROPY AND FREE ENERGY
            loss, predictions = loss_fn(
                quantity_trajs, weights, entropy, free_energy)
            return loss, (predictions, entropy, free_energy)

        # statepoint_grad_fn = jax.jit(value_and_grad(difftre_loss, has_aux=True))  #donate_argnums=1)
        statepoint_grad_fn = jit(value_and_grad(difftre_loss, has_aux=True))

        ## AGAIN THE SAME, BUT ADAPTED TO RETURN ENTROPY AND FREE ENERGY
        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state, species)
            outputs, grad = statepoint_grad_fn(params, traj_state)
            loss_val, (predictions, entropy, free_energy) = outputs
            return traj_state, grad, loss_val, predictions, entropy, free_energy


        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point
        ##OPTIONAL TO DISPLAY TARGET FREE ENERGY WHEN TRAINING
        self._targets[key] = targets['free_energy_difference']['target']


        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

        # self._one_graph_bool = True


    # def get_gpu_memory(self):
    #     """Get the current gpu memory usage in GB."""
    #
    #     try:
    #         result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    #                                 capture_output=True, text=True, check=True)
    #         gpu_stats = result.stdout.strip().split('\n')
    #         return gpu_stats[1]
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error: {e}")

    ## UPDATE FUNCTION ADAPTED TO THE EXTRA OUTPUTS OF THE LOSS FN
    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts
        grads, losses = [], []

        # losses = 0.0
        # grads = None

        for sim_key in batch:

            grad_fn = self.grad_fns[sim_key]
            ## EXTRA ENTROPY AND FREE ENERGY OUTPUTS
            (new_traj_state, curr_grad, loss_val, state_point_predictions,
             statepoint_entropy, statepoint_free_energy) = grad_fn(
                self.params, self.trajectory_states[sim_key])

            state_point_predictions.update({
                "free_energy_difference": statepoint_free_energy
            })

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = state_point_predictions
            # self.predictions[sim_key][self._epoch] = tree_util.tree_map(
            #     onp.asarray, state_point_predictions)
            # losses += loss_val
            # if grads is None:
            #     grads = curr_grad
            # else:
            #     grads = util.tree_sum(grads, curr_grad)

            G = nx.Graph()
            edges_precomputed = [(int(new_traj_state.sim_state[1].idx[0][count]), int(new_traj_state.sim_state[1].idx[1][count])) for count in
                                 range(onp.sum(new_traj_state.sim_state[1].idx[0] != new_traj_state.sim_state[0].position.shape[0]))]
            G.add_nodes_from(onp.arange(new_traj_state.sim_state[0].position.shape[0]))
            G.add_edges_from(edges_precomputed)
            check_if_one_graph = nx.is_connected(G)

            # Use graph to check whether all nodes are connected

            # if not check_if_one_graph:
            #     self._one_graph_bool = False    # Assures that if it's false once it is always false
            # if check_if_one_graph and self._one_graph_bool:
            if check_if_one_graph:
                grads.append(curr_grad)
                losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self._diverged = True  # ends training
                break

        # self.batch_losses.append(onp.asarray(losses / self.sim_batch_size))
        # batch_grad = tree_util.tree_map(lambda x: x / self.sim_batch_size, grads)
        # if check_if_one_graph and self._one_graph_bool:
        if check_if_one_graph:
            self.batch_losses.append(sum(losses) / self.sim_batch_size)
            batch_grad = util.tree_mean(grads)
            self._step_optimizer(batch_grad)
            self.gradient_norm_history.append(util.tree_norm(batch_grad))
        else:
            self.batch_losses.append(0)
            self.gradient_norm_history.append(0)
        ## ADDED/USE UPDATE COUNTER AS USEFUL FOR DEBUGGING
        # batch_norm = util.tree_norm(batch_grad)
        # self.gradient_norm_history.append(onp.asarray(batch_norm))

        # del grads, loss_val
        # gc.collect()

        self.update_counter += 1
        print('update_counter: ', self.update_counter)

        # Debug memory
        # jax.clear_caches()
        # memory = self.get_gpu_memory()
        # print('GPU memory used [GB]: ', memory)
        # self.gpu_memory_list.append(memory)


    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(f'\nEpoch {self._epoch}: Epoch loss = {epoch_loss:.5f}, Gradient '
              f'norm: {self.gradient_norm_history[-1]}, '
              f'Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        # print last scalar predictions
        for statepoint, prediction_series in self.predictions.items():
            last_predictions = prediction_series[self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    print(f'Statepoint {statepoint}: Predicted {quantity}:'
                          f' {value}', 'Target :', self._targets[statepoint])

        self._converged = self.early_stop.early_stopping(epoch_loss, thresh,
                                                         self.params)

    @property
    def best_params(self):
        return self.early_stop.best_params

    def move_to_device(self):
        super().move_to_device()
        self.early_stop.move_to_device()

    def init_traj(self, statepoint):
        return self.trajectory_states[statepoint]

    def delete_statepoint_count(self):
        self.n_statepoints -= 1

    def reset_params(self, new_params):
        self.params = new_params

    def return_FE_values(self):
        return self.init_FE_values

class DifftreActive(util.TrainerInterface):
    """Active learning of state-transferable potentials from experimental data
     via DiffTRe.

     The input trainer can be pre-trained or freshly initialized. Pre-training
     usually comes with the advantage that the initial training from random
     parameters is usually the most unstable one. Hence, special care can be
     taken such as training on NVT initially to fix the pressure and swapping
     to NPT afterwards. This active learning trainer then takes care of learning
      statepoint transferability.
     """
    def __init__(self, trainer, checkpoint_folder='Checkpoints',
                 energy_fn_template=None):
        checkpoint_path = 'output/difftre_active/' + str(checkpoint_folder)
        super().__init__(checkpoint_path, energy_fn_template)
        self.trainer = trainer
        # other inits

    def add_statepoint(self, *args, **kwargs):
        """Add another statepoint to the target state points.

        Predominantly used to add statepoints with more / different targets
        not covered in  the on-the-fly tepoint addition, e.g. for an extensive
        initial statepoint. Please refer to :obj:'Difftre.add_statepoint
        <chemtrain.trainers.Difftre.add_statepoint>' for the full documentation.
        """
        self.trainer.add_statepoint(*args, **kwargs)

    def train(self, max_new_statepoints=100):
        for added_statepoints in range(max_new_statepoints):
            accuracy_met = False
            if accuracy_met:
                print('Visited state space covered with accuracy target met.')
                break

            # checkpoint: call checkpoint of trainer
        else:
            warnings.warn('Maximum number of added statepoints added without '
                          'reaching target accuracy over visited state space.')

    @property
    def params(self):
        return self.trainer.params

    @params.setter
    def params(self, loaded_params):
        self.trainer.params = loaded_params


class RelativeEntropy(reweighting.PropagationBase):
    """Trainer for relative entropy minimization."""
    def __init__(self, init_params, optimizer,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """
        Initializes a relative entropy trainer instance.

        Uses first order method optimizer as Hessian is very expensive
        for neural networks. Both reweighting and the gradient formula
        currently assume a NVT ensemble.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the gradient norm
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average gradient norm across the batch.
                                   For a single state point, both are
                                   equivalent. A criterion based on the rolling
                                   standard deviation 'std' might be implemented
                                   in the future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        checkpoint_path = 'output/rel_entropy/' + str(checkpoint_folder)
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, energy_fn_template)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}

        self.early_stop = max_likelihood.EarlyStopping(self.params,
                                                       convergence_criterion)

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = numpy_loader.NumpyDataLoader(R=reference_data,
                                                        copy=False)
        init_reference_batch, get_ref_batch, _ = data.random_reference_data(
            reference_loader, batch_cache, reference_batch_size)
        init_reference_batch_state = init_reference_batch(shuffle=True)
        self.data_states[key] = init_reference_batch_state
        return get_ref_batch

    def add_statepoint(self, reference_data, energy_fn_template,
                       simulator_template, neighbor_fn, timings, kbt,
                       reference_state, reference_batch_size=None,
                       batch_cache=1, initialize_traj=True, set_key=None,
                       vmap_batch=10):
        """
        Adds a state point to the pool of simulations.

        As each reference dataset / trajectory corresponds to a single
        state point, we initialize the dataloader together with the
        simulation.

        Currently only supports NVT simulations.

        Args:
            reference_data: De-correlated reference trajectory
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                                  trajectory. If None, will use the same number
                                  of snapshots as generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                         minimize host-device communication. Make sure the
                         cached data size does not exceed the full dataset size.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                     By default, uses the index of the sequance statepoints are
                     added, i.e. self.trajectory_states[0] for the first added
                     statepoint. Can be used for changing the timings of the
                     simulation during training.
            vmap_batch: Batch size of vmapping of per-snapshot energy and
                        gradient calculation.
        """
        if reference_batch_size is None:
            print('No reference batch size provided. Using number of generated'
                  ' CG snapshots by default.')
            states_per_traj = jnp.size(timings.t_production_start)
            if reference_state[0].position.ndim > 2:
                n_trajctories = reference_state[0].position.shape[0]
                reference_batch_size = n_trajctories * states_per_traj
            else:
                reference_batch_size = states_per_traj

        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           set_key,
                                                           vmap_batch,
                                                           initialize_traj)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        grad_fn = reweighting.init_rel_entropy_gradient(
            energy_fn_template, weights_fn, kbt, vmap_batch)

        def propagation_and_grad(params, traj_state, batch_state):
            """Propagates the trajectory, if necessary, and computes the
            gradient via the relative entropy formalism.
            """
            traj_state = propagate(params, traj_state)
            new_batch_state, reference_batch = reference_dataloader(batch_state)
            reference_positions = reference_batch['R']
            grad = grad_fn(params, traj_state, reference_positions)
            return traj_state, grad, new_batch_state

        self.grad_fns[key] = propagation_and_grad

    def _update(self, batch):
        """Updates the potential using the gradient from relative entropy."""
        grads = []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]

            self.trajectory_states[sim_key], curr_grad, \
            self.data_states[sim_key] = grad_fn(self.params,
                                                self.trajectory_states[sim_key],
                                                self.data_states[sim_key])
            grads.append(curr_grad)

        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)
        self.gradient_norm_history.append(util.tree_norm(batch_grad))

    def _evaluate_convergence(self, duration, thresh):
        curr_grad_norm = self.gradient_norm_history[-1]
        print(f'\nEpoch {self._epoch}: Gradient norm: '
              f'{curr_grad_norm}, Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        self._converged = self.early_stop.early_stopping(curr_grad_norm, thresh,
                                                         save_best_params=False)


class SGMCForceMatching(probabilistic.ProbabilisticFMTrainerTemplate):
    """Trainer for stochastic gradient Markov-chain Monte Carlo training
    based on force-matching.

    init_samples: A list, possibly of size 1, of sets of initial MCMC samples,
     where each spawns a dedicated MCMC chain,
    """
    def __init__(self, sgmc_solver, init_samples, val_dataloader=None,
                 energy_fn_template=None):
        # TODO: Where does alias.py get checkpoint_path info?
        super().__init__(None, energy_fn_template)
        self._params = [init_sample['params'] for init_sample in init_samples]
        self.sgmcmc_run_fn = sgmc_solver
        self.init_samples = init_samples

        # TODO use val dataloader to compute posterior predictive p value or
        #  other convergence metric. In ProbabilisticFMTrainerTemplate??

        # TODO also use test_set?

    def train(self, iterations):
        """Training of any trainer should start by calling train."""
        self.results = self.sgmcmc_run_fn(*self.init_samples,
                                          iterations=iterations)

    @property
    def params(self):
        assert len(self.results) == 1, 'Not implemented for multiple chains'
        # TODO: Need to edit this for multiple chains!
        params = []
        for chain in self.results:
            params = chain['samples']['variables']['params']
        return params

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError('Setting params seems not meaningful in'
                                  ' the case of SG-MCMC samplers.')

    @property
    def list_of_params(self):
        return util.tree_unstack(self.params)
    # TODO override save functions such that only saving parameters is allowed
    #  - or whatever checkpointing jax-sgmc supports (or does checkpointing work
    #  with more liberal coax._jit?)


# TODO adjust to new blackjax interface, then allow newer version
class NUTSForceMatching(probabilistic.MCMCForceMatchingTemplate):
    """Trainer that samples from the posterior distribution of energy_params via
    the No-U-Turn Sampler (NUTS), based on a force-matching formulation.
    """
    def __init__(self, prior, likelihood, train_loader, init_sample,
                 batch_cache=1, batch_size=1, val_loader=None,
                 warmup_steps=1000, step_size=None,
                 inv_mass_matrix=None, checkpoint_folder='Checkpoints',
                 ref_energy_fn_template=None, init_prng_key=random.PRNGKey(0)):
        checkpoint_path = 'output/NUTS/' + str(checkpoint_folder)

        log_posterior_fn = probabilistic.init_log_posterior_fn(
            likelihood, prior, train_loader, batch_size, batch_cache
        )
        init_state = nuts.new_state(init_sample, log_posterior_fn)

        if step_size is None or inv_mass_matrix is None:
            def warmup_gen_fn(step, inverse_mass_matrix):
                return nuts.kernel(log_posterior_fn, step,
                                   inverse_mass_matrix)

            init_state, (step_size, inv_mass_matrix), info = stan_warmup.run(
                init_prng_key, warmup_gen_fn, init_state, warmup_steps)
            print('Finished warmup.\n', info)

        kernel = nuts.kernel(log_posterior_fn, step_size,
                             inv_mass_matrix)
        super().__init__(init_state, kernel, checkpoint_path, val_loader,
                         ref_energy_fn_template)


class EnsembleOfModels(probabilistic.ProbabilisticFMTrainerTemplate):
    """Train an ensemble of models by starting optimization from different
    initial parameter sets, for use in uncertainty quantification applications.
    """
    def __init__(self, trainers, ref_energy_fn_template=None):
        super().__init__(None, ref_energy_fn_template)
        self.trainers = trainers

    def train(self, *args, **kwargs):
        for i, trainer in enumerate(self.trainers):
            print(f'---------Starting trainer {i}-----------')
            trainer.train(*args, **kwargs)
        print('Finished training all models.')

    @property
    def params(self):
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        for i, params in enumerate(loaded_params):
            self.trainers[i].params = params

    @property
    def list_of_params(self):
        params = []
        for trainer in self.trainers:
            if hasattr(trainer, 'best_params'):
                params.append(trainer.best_params)
            else:
                params.append(trainer.params)
        return params


class wrapper_ForceMatching():

    def __init__(self, trainerFM):
        self._trainer = trainerFM


    def train(self, num_epochs, save_path, path_to_project):
        num_epochs = int(num_epochs)
        for i in range(num_epochs):
            self._trainer.train(1, checkpoint_freq=None)

            best_params_path = path_to_project + 'FreeEnergy_Publication/examples/FreeEnergyScripts/savedTrainers/'+str(save_path)+'_epoch_'+str(i+1)+'_Params.pkl'
            with open(best_params_path, 'wb') as pickle_file:
                pickle.dump(self._trainer.params, pickle_file)

        last_params_path = path_to_project + 'FreeEnergy_Publication/examples/FreeEnergyScripts/savedTrainers/' + str(
            save_path) + '_lastParams.pkl'
        with open(last_params_path, 'wb') as pickle_file:
            pickle.dump(self._trainer.params, pickle_file)

        train_loss_path = path_to_project + 'FreeEnergy_Publication/examples/FreeEnergyScripts/savedTrainers/'+str(save_path)+'_trainLoss.pkl'
        val_loss_path = path_to_project + 'FreeEnergy_Publication/examples/FreeEnergyScripts/savedTrainers/' + str(save_path) + '_valLoss.pkl'
        onp.save(train_loss_path, self._trainer.train_losses)
        onp.save(val_loss_path, self._trainer.val_losses)


## TRAINER WRAPPER TO CIRCUMVENT FREEZING WHEN SAVING ENERGY PARAMS IN FRONT-END
class wrapper_trainer_HFE():

    def __init__(self, trainer):
        self._trainer = trainer

    def train(self, num_epochs, save_checkpoint, save_epochs):
        for k in range(num_epochs):
            self._trainer.train(1)
            # onp.save('memory_usage/memory_usage_epoch_' + str(k+1) + '.npy', onp.array(self._trainer.gpu_memory_list))

            if k in save_epochs:
                self._trainer.save_energy_params('checkpoints/' + save_checkpoint + f'_epoch{k}.pkl', '.pkl')
                predicted_quantities = self._trainer.predictions
                statepoint_dict = {}
                for i in range(len(predicted_quantities)):
                    pred_dict = predicted_quantities[i][k]
                    FE = pred_dict['free_energy_difference']
                    pred_list = [FE.tolist()]
                    statepoint_dict[f'{i}'] = pred_list
                loss = jnp.array(self._trainer.batch_losses)
                loss_list = loss.tolist()

                with open('checkpoints/' + save_checkpoint + f'_pred_dict_epoch{k}.pkl', 'wb') as f:
                    pickle.dump(statepoint_dict, f)

                with open('checkpoints/' + save_checkpoint + f'_loss_epoch{k}.pkl', 'wb') as f:
                    pickle.dump(loss_list, f)

        return
