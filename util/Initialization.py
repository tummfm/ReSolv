"""A collection of functions to initialize Jax, M.D. simulations."""
from functools import partial

import chex
import haiku as hk
from jax import random, vmap, nn, numpy as jnp, lax
from jax_md import util, simulate, partition, space, energy
import numpy as onp
from scipy import interpolate as sci_interpolate

from chemtrain import traj_quantity, layers, neural_networks, dropout
from chemtrain.jax_md_mod import custom_energy, custom_space, custom_quantity

import chemtrain.amber_utils_qm7x as au
import chemtrain.nequip_amber_utils_qm7x as au_nequip
# For Nequip
import jax.nn
from ml_collections import ConfigDict

import pickle
from jax.nn import one_hot


Array = util.Array


@chex.dataclass
class InitializationClass:
    """A dataclass containing initialization information.

    Notes:
      careful: dataclasses.astuple(InitializationClass) sometimes
      changes type from jnp.Array to onp.ndarray

    Attributes:
        r_init: Initial Positions
        box: Simulation box size
        kbT: Target thermostat temperature times Boltzmann constant
        mass: Particle masses
        dt: Time step size
        species: Species index for each particle
        ref_press: Target pressure for barostat
        temperature: Thermostat temperature; only used for computation of
                     thermal expansion coefficient and heat capacity
    """
    r_init: Array
    box: Array
    kbt: float
    masses: Array
    dt: float
    species: Array = None
    ref_press: float = 1.
    temperature: float = None


def select_target_rdf(target_rdf, rdf_start=0., nbins=300):
    if target_rdf == 'LJ':
        reference_rdf = util.f32(onp.loadtxt('data/LJ_reference_RDF.csv'))
        rdf_cut = 1.5
        raise NotImplementedError
    elif target_rdf == 'SPC':
        reference_rdf = onp.loadtxt('data/water_models/SPC_955_RDF.csv')
        rdf_cut = 1.0
    elif target_rdf == 'SPC_FW':
        reference_rdf = onp.loadtxt('data/water_models/SPC_FW_RDF.csv')
        rdf_cut = 1.0
    elif target_rdf == 'TIP4P/2005':
        reference_rdf = onp.loadtxt('data/water_models/'
                                    'TIP4P-2005_300_COM_RDF.csv')
        rdf_cut = 0.85
    elif target_rdf == 'Water_Ox':
        reference_rdf = onp.loadtxt('data/experimental/O_O_RDF.csv')
        rdf_cut = 1.0
    else:
        raise ValueError(f'The reference rdf {target_rdf} is not implemented.')

    rdf_bin_centers, rdf_bin_boundaries, sigma_rdf = \
        custom_quantity.rdf_discretization(rdf_cut, nbins, rdf_start)
    rdf_spline = sci_interpolate.interp1d(reference_rdf[:, 0],
                                          reference_rdf[:, 1], kind='cubic')
    reference_rdf = util.f32(rdf_spline(rdf_bin_centers))
    rdf_struct = custom_quantity.RDFParams(reference_rdf, rdf_bin_centers,
                                           rdf_bin_boundaries, sigma_rdf)
    return rdf_struct


def select_target_adf(target_adf, r_outer, r_inner=0., nbins_theta=150):
    if target_adf == 'Water_Ox':
        reference_adf = onp.loadtxt('data/experimental/O_O_O_ADF.csv')
    elif target_adf == 'TIP4P/2005':
        reference_adf = onp.loadtxt('data/water_models/'
                                    'TIP4P-2005_150_COM_ADF.csv')
    else:
        raise ValueError(f'The reference adf {target_adf} is not implemented.')

    adf_bin_centers, sigma_adf = custom_quantity.adf_discretization(nbins_theta)

    adf_spline = sci_interpolate.interp1d(reference_adf[:, 0],
                                          reference_adf[:, 1], kind='cubic')
    reference_adf = util.f32(adf_spline(adf_bin_centers))

    adf_struct = custom_quantity.ADFParams(reference_adf, adf_bin_centers,
                                           sigma_adf, r_outer, r_inner)
    return adf_struct


def select_target_tcf(target_tcf, tcf_cut, tcf_start=0.2, nbins=30):
    if target_tcf == 'TIP4P/2005':
        if tcf_cut == 0.5:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut05.npy')
            dx_bin = 0.3 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.5 - dx_bin/2., 50)
        elif tcf_cut == 0.6:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut06.npy')
            dx_bin = 0.4 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.6 - dx_bin/2., 50)
        elif tcf_cut == 0.8:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut08.npy')
            dx_bin = 0.6 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.8 - dx_bin/2., 50)
        else:
            raise ValueError(f'The cutoff {tcf_cut} is not implemented.')
    else:
        raise ValueError(f'The reference tcf {target_tcf} is not implemented.')

    (sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers,
     tcf_z_bin_centers) = custom_quantity.tcf_discretization(tcf_cut, nbins,
                                                             tcf_start)

    equilateral = onp.diagonal(onp.diagonal(reference_tcf))
    tcf_spline = sci_interpolate.interp1d(bins_centers, equilateral,
                                          kind='cubic')
    reference_tcf = util.f32(tcf_spline(tcf_x_binx_centers[0, :, 0]))
    tcf_struct = custom_quantity.TCFParams(
        reference_tcf, sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers,
        tcf_z_bin_centers
    )
    return tcf_struct


def prior_potential(prior_fns, pos, neighbor, **dynamic_kwargs):
    """Evaluates the prior potential for a given snapshot."""
    sum_priors = 0.
    if prior_fns is not None:
        for key in prior_fns:
            sum_priors += prior_fns[key](pos, neighbor=neighbor,
                                         **dynamic_kwargs)
    return sum_priors


def select_priors(displacement, prior_constants, prior_idxs, kbt=None):
    """Build prior potential from combination of classical potentials."""
    prior_fns = {}
    if 'bond' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for bond prior.'
        bond_mean, bond_variance = prior_constants['bond']
        bonds = prior_idxs['bond']
        prior_fns['bond'] = energy.simple_spring_bond(
            displacement, bonds, length=bond_mean, epsilon=kbt / bond_variance)

    if 'angle' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for angle prior.'
        angle_mean, angle_variance = prior_constants['angle']
        angles = prior_idxs['angle']
        prior_fns['angle'] = custom_energy.harmonic_angle(
            displacement, angles, angle_mean, angle_variance, kbt)

    if 'LJ' in prior_constants:
        lj_sigma, lj_epsilon = prior_constants['LJ']
        lj_idxs = prior_idxs['LJ']
        prior_fns['LJ'] = custom_energy.lennard_jones_nonbond(
            displacement, lj_idxs, lj_sigma, lj_epsilon)

    if 'repulsive' in prior_constants:
        re_sigma, re_epsilon, re_cut = prior_constants['repulsive']
        prior_fns['repulsive'] = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=re_sigma, epsilon=re_epsilon, exp=12,
            initialize_neighbor_list=False, r_onset=0.9 * re_cut,
            r_cutoff=re_cut)

    if 'dihedral' in prior_constants:
        dih_phase, dih_constant, dih_n = prior_constants['dihedral']
        dihdral_idxs = prior_idxs['dihedral']
        prior_fns['dihedral'] = custom_energy.periodic_dihedral(
            displacement, dihdral_idxs, dih_phase, dih_constant, dih_n)

    if 'repulsive_nonbonded' in prior_constants:
        # only repulsive part of LJ via idxs instead of nbrs list
        ren_sigma, ren_epsilon = prior_constants['repulsive_nonbonded']
        ren_idxs = prior_idxs['repulsive_nonbonded']
        prior_fns['repulsive_1_4'] = custom_energy.generic_repulsion_nonbond(
            displacement, ren_idxs, sigma=ren_sigma, epsilon=ren_epsilon, exp=6)

    return prior_fns


def select_protein(protein, prior_list):
    idxs = {}
    constants = {}
    if protein == 'heavy_alanine_dipeptide':
        print('Distinguishing different C_Hx atoms')
        species = jnp.array([6, 1, 8, 7, 2, 6, 1, 8, 7, 6])
        if 'bond' in prior_list:
            bond_mean = onp.load('data/prior/Alanine_dipeptide_heavy_eq_bond'
                                 '_length.npy')
            bond_variance = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                     '_bond_variance.npy')
            bond_idxs = onp.array([[0, 1],
                                   [1, 2],
                                   [1, 3],
                                   [4, 6],
                                   [6, 7],
                                   [4, 5],
                                   [3, 4],
                                   [6, 8],
                                   [8, 9]])
            idxs['bond'] = bond_idxs
            constants['bond'] = (bond_mean, bond_variance)

        if 'angle' in prior_list:
            angle_mean = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                  '_angle.npy')
            angle_variance = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                      '_angle_variance.npy')
            angle_idxs = onp.array([[0, 1, 2],
                                    [0, 1, 3],
                                    [2, 1, 3],
                                    [1, 3, 4],
                                    [3, 4, 5],
                                    [3, 4, 6],
                                    [5, 4, 6],
                                    [4, 6, 7],
                                    [4, 6, 8],
                                    [7, 6, 8],
                                    [6, 8, 9]])
            idxs['angle'] = angle_idxs
            constants['angle'] = (angle_mean, angle_variance)

        if 'LJ' in prior_list:
            lj_sigma = onp.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            lj_epsilon = onp.load('data/prior/Alanine_dipeptide_heavy_'
                                  'epsilon.npy')
            lj_idxs = onp.array([[0, 5],
                                 [0, 6],
                                 [0, 7],
                                 [0, 8],
                                 [0, 9],
                                 [1, 7],
                                 [1, 8],
                                 [1, 9],
                                 [2, 5],
                                 [2, 6],
                                 [2, 7],
                                 [2, 8],
                                 [2, 9],
                                 [3, 9],
                                 [5, 9]])
            idxs['LJ'] = lj_idxs
            constants['LJ'] = (lj_sigma, lj_epsilon)

        if 'dihedral' in prior_list:
            dihedral_phase = onp.load('data/prior/Alanine_dipeptide_heavy_'
                                      'dihedral_phase.npy')
            dihedral_constant = onp.load('data/prior/Alanine_dipeptide_heavy'
                                         '_dihedral_constant.npy')
            dihedral_n = onp.load('data/prior/Alanine_dipeptide_heavy_dihedral'
                                  '_multiplicity.npy')

            dihedral_idxs = onp.array([[1, 3, 4, 6],
                                       [3, 4, 6, 8],
                                       [0, 1, 3, 4],
                                       [2, 1, 3, 4],
                                       [1, 3, 4, 5],
                                       [5, 4, 6, 8],
                                       [4, 6, 8, 9],
                                       [7, 6, 8, 9]])
            idxs['dihedral'] = dihedral_idxs
            constants['dihedral'] = (dihedral_phase, dihedral_constant,
                                     dihedral_n)

        if 'repulsive_nonbonded' in prior_list:
            # repulsive part of the LJ
            if 'LJ' in prior_list:
                raise ValueError('Not sensible to have LJ and repulsive part of'
                                 ' LJ together. Choose one.')
            ren_sigma = onp.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            ren_epsilon = onp.load('data/prior/Alanine_dipeptide'
                                   '_heavy_epsilon.npy')
            ren_idxs = onp.array([[0, 5],
                                  [0, 6],
                                  [0, 7],
                                  [0, 8],
                                  [0, 9],
                                  [1, 7],
                                  [1, 8],
                                  [1, 9],
                                  [2, 5],
                                  [2, 6],
                                  [2, 7],
                                  [2, 8],
                                  [2, 9],
                                  [3, 9],
                                  [5, 9]])
            idxs['repulsive_nonbonded'] = ren_idxs
            constants['repulsive_nonbonded'] = (ren_sigma, ren_epsilon)
    else:
        raise ValueError(f'The protein {protein} is not implemented.')
    return species, idxs, constants


def build_quantity_dict(pos_init, box_tensor, displacement, energy_fn_template,
                        nbrs, target_dict, init_class):
    targets = {}
    compute_fns = {}
    kj_mol_nm3_to_bar = 16.6054

    if 'kappa' in target_dict or 'alpha' in target_dict or 'cp' in target_dict:
        compute_fns['volume'] = custom_quantity.volume_npt
        if 'alpha' in target_dict or 'cp' in target_dict:
            compute_fns['energy'] = custom_quantity.energy_wrapper(
                energy_fn_template)

    if 'rdf' in target_dict:
        rdf_struct = target_dict['rdf']
        rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)
        rdf_dict = {'target': rdf_struct.reference, 'gamma': 1.,
                    'traj_fn': traj_quantity.init_traj_mean_fn('rdf')}
        targets['rdf'] = rdf_dict
        compute_fns['rdf'] = rdf_fn

    if 'adf' in target_dict:
        adf_struct = target_dict['adf']
        adf_fn = custom_quantity.init_adf_nbrs(
            displacement, adf_struct, smoothing_dr=0.01, r_init=pos_init,
            nbrs_init=nbrs)
        adf_target_dict = {'target': adf_struct.reference, 'gamma': 1.,
                           'traj_fn': traj_quantity.init_traj_mean_fn('adf')}
        targets['adf'] = adf_target_dict
        compute_fns['adf'] = adf_fn

    if 'tcf' in target_dict:
        tcf_struct = target_dict['tcf']
        tcf_fn = custom_quantity.init_tcf_nbrs(displacement, tcf_struct,
                                               box_tensor, nbrs_init=nbrs,
                                               batch_size=1000)
        tcf_target_dict = {'target': tcf_struct.reference, 'gamma': 1.,
                           'traj_fn': traj_quantity.init_traj_mean_fn('tcf')}
        targets['tcf'] = tcf_target_dict
        compute_fns['tcf'] = tcf_fn

    if 'pressure' in target_dict:
        pressure_fn = custom_quantity.init_pressure(energy_fn_template,
                                                    box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure'], 'gamma': 1.e-7,
            'traj_fn': traj_quantity.init_traj_mean_fn('pressure')}
        targets['pressure'] = pressure_target_dict
        compute_fns['pressure'] = pressure_fn

    if 'pressure_tensor' in target_dict:
        pressure_fn = custom_quantity.init_virial_stress_tensor(
            energy_fn_template, box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure_tensor'], 'gamma': 1.e-7,
            'traj_fn': traj_quantity.init_traj_mean_fn('pressure_tensor')}
        targets['pressure_tensor'] = pressure_target_dict
        compute_fns['pressure_tensor'] = pressure_fn

    if 'density' in target_dict:
        density_dict = {
            'target': target_dict['density'], 'gamma': 1.e-3,  # 1.e-5
            'traj_fn': traj_quantity.init_traj_mean_fn('density')
        }
        targets['density'] = density_dict
        compute_fns['density'] = custom_quantity.density

    if 'kappa' in target_dict:
        def compress_traj_fn(quantity_trajs):
            volume_traj = quantity_trajs['volume']
            kappa = traj_quantity.isothermal_compressibility_npt(volume_traj,
                                                                 init_class.kbt)
            return kappa

        comp_dict = {
            'target': target_dict['kappa'],
            'gamma': 1. / (5.e-5 * kj_mol_nm3_to_bar),
            'traj_fn': compress_traj_fn
        }
        targets['kappa'] = comp_dict

    if 'alpha' in target_dict:
        def thermo_expansion_traj_fn(quantity_trajs):
            alpha = traj_quantity.thermal_expansion_coefficient_npt(
                quantity_trajs['volume'], quantity_trajs['energy'],
                init_class.temperature, init_class.kbt, init_class.ref_press)
            return alpha

        alpha_dict = {
            'target': target_dict['alpha'], 'gamma': 1.e4,
            'traj_fn': thermo_expansion_traj_fn
        }
        targets['alpha'] = alpha_dict

    if 'cp' in target_dict:
        n_particles, dim = pos_init.shape
        # assuming no reduction, e.g. due to rigid bonds
        n_dof = dim * n_particles

        def cp_traj_fn(quantity_trajs):
            cp = traj_quantity.specific_heat_capacity_npt(
                quantity_trajs['volume'], quantity_trajs['energy'],
                init_class.temperature, init_class.kbt, init_class.ref_press,
                n_dof)
            return cp

        cp_dict = {
            'target': target_dict['cp'], 'gamma': 10.,
            'traj_fn': cp_traj_fn
        }
        targets['cp'] = cp_dict

    if 'free_energy_difference' in target_dict:
        targets['free_energy_difference'] = {'target': target_dict['free_energy_difference'], 'gamma': 1.}

    return compute_fns, targets


def default_x_vals(r_cut, delta_cut):
    return jnp.linspace(0.05, r_cut + delta_cut, 100, dtype=jnp.float32)


def select_model(model, init_pos, displacement, box, model_init_key, kbt=None,
                 species=None, x_vals=None, fractional=True,
                 kbt_dependent=False, prior_constants=None, prior_idxs=None,
                 dropout_init_seed=None, max_edges=None, max_angles=None, mol_id_data=None,
                 id_mapping_rev=None, vac_model_path=None, **energy_kwargs):
    if model == 'LJ':
        r_cut = 0.9
        init_params = jnp.array([0.2, 1.2], dtype=jnp.float32)  # initial guess
        lj_neighbor_energy = partial(
            custom_energy.customn_lennard_jones_neighbor_list, displacement,
            box, r_onset=0.8, r_cutoff=r_cut, dr_threshold=0.2,
            capacity_multiplier=1.25, fractional=fractional)
        neighbor_fn, _ = lj_neighbor_energy(sigma=init_params[0],
                                            epsilon=init_params[1])
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        def energy_fn_template(energy_params):
            # we only need to re-create energy_fn, neighbor function is re-used
            lj_energy = lj_neighbor_energy(
                sigma=energy_params[0], epsilon=energy_params[1],
                initialize_neighbor_list=False)
            return lj_energy

    elif model == 'Tabulated':
        # TODO: change initial guess to generic LJ or random initialization
        # TODO adjust to new prior interface
        r_cut = 0.9
        delta_cut = 0.1
        if x_vals is None:
            x_vals = default_x_vals(r_cut, delta_cut)

        # load PMF initial guess
        # pmf_init = False  # for IBI
        pmf_init = False
        if pmf_init:
            # table_loc = 'data/tabulated_potentials/CG_potential_SPC_955.csv'
            table_loc = 'data/tabulated_potentials/IBI_initial_guess.csv'
            tabulated_array = onp.loadtxt(table_loc)
            # compute tabulated values at spline support points
            u_init_int = sci_interpolate.interp1d(tabulated_array[:, 0],
                                        tabulated_array[:, 1], kind='cubic')
            init_params = jnp.array(u_init_int(x_vals), dtype=jnp.float32)
        else:
            # random initialisation + prior
            init_params = 0.1 * random.normal(model_init_key, x_vals.shape)
            init_params = jnp.array(init_params, dtype=jnp.float32)
            prior_fn = custom_energy.generic_repulsion_neighborlist(
                displacement, sigma=0.3165, epsilon=1., exp=12,
                initialize_neighbor_list=False, r_onset=0.9 * r_cut,
                r_cutoff=r_cut)

        tabulated_energy = partial(
            custom_energy.tabulated_neighbor_list, displacement, x_vals,
            box_size=box, r_onset=(r_cut - 0.2), r_cutoff=r_cut,
            dr_threshold=0.05, capacity_multiplier=1.25
        )
        neighbor_fn, _ = tabulated_energy(init_params)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        if pmf_init:
            def energy_fn_template(energy_params):
                tab_energy = tabulated_energy(energy_params,
                                              initialize_neighbor_list=False)
                return tab_energy
        else:  # with prior
            def energy_fn_template(energy_params):
                tab_energy = tabulated_energy(energy_params,
                                              initialize_neighbor_list=False)

                def energy_fn(pos, neighbor, **dynamic_kwargs):
                    return (tab_energy(pos, neighbor, **dynamic_kwargs)
                            + prior_fn(pos, neighbor=neighbor, **dynamic_kwargs)
                            )
                return energy_fn

    elif model == 'PairNN':
        # TODO adjust to new prior interface
        r_cut = 3.  # 3 sigma in LJ units
        hidden_layers = [64, 64]  # with 32 higher best force error

        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.5,
                                              capacity_multiplier=1.5,
                                              fractional_coordinates=fractional)
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)
        prior_fn = custom_energy.generic_repulsion_neighborlist(
            displacement,
            sigma=0.7,
            epsilon=1.,
            exp=12,
            initialize_neighbor_list=False,
            r_onset=0.9 * r_cut,
            r_cutoff=r_cut
        )

        init_fn, pair_nn_energy = neural_networks.pair_interaction_nn(
            displacement, r_cut, hidden_layers)
        if isinstance(model_init_key, list):
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos, neighbor=nbrs_init)

        def energy_fn_template(energy_params):
            pair_nn_energy_fix = partial(pair_nn_energy, energy_params,
                                         species=species)

            def energy_fn(pos, neighbor, **dynamic_kwargs):
                return (pair_nn_energy_fix(pos, neighbor, **dynamic_kwargs) +
                        prior_fn(pos, neighbor=neighbor, **dynamic_kwargs))
            return energy_fn

    elif model in ['CGDimeNet', 'NequIP']:
        # r_cut = 0.5
        r_cut = 5.0
        n_species = 100
        neighbor_capacity_multiple = 1.5
        extra_capacity = 0
        dr_thresh = 0.05

        if prior_constants is not None:
            prior_fns = select_priors(displacement, prior_constants, prior_idxs,
                                      kbt)
            print('Using the following priors:')
            [print(key) for key in prior_fns]
        else:
            print('Using no priors')
            prior_fns = None

        if model == 'CGDimeNet':
            mlp_init = {
                'b_init': hk.initializers.Constant(0.),
                'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
            }

            neighbor_fn = partition.neighbor_list(
                displacement, box, r_cut, dr_threshold=dr_thresh,
                capacity_multiplier=neighbor_capacity_multiple,
                fractional_coordinates=fractional, disable_cell_list=True)

            nbrs_init = neighbor_fn.allocate(init_pos,
                                             extra_capacity=extra_capacity)

            dropout_mode = {'output': 0.1, 'interaction': 0.1, 'embedding': 0.1}

            init_fn, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
                displacement, r_cut, n_species, init_pos, nbrs_init,
                kbt_dependent=kbt_dependent, embed_size=32,
                init_kwargs=mlp_init, dropout_mode=dropout_mode
            )

            if isinstance(model_init_key, list):
                # ensemble of neural networks not needed together with dropout
                init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                       species=species, **energy_kwargs)
                               for key in model_init_key]
            else:
                if dropout_init_seed is None:
                    init_params = init_fn(model_init_key, init_pos,
                                          neighbor=nbrs_init,
                                          species=species, **energy_kwargs)
                else:
                    dropout_init_key = random.PRNGKey(dropout_init_seed)
                    init_params = init_fn(model_init_key, init_pos,
                                          neighbor=nbrs_init, species=species,
                                          dropout_key=dropout_init_key,
                                          **energy_kwargs)
                    init_params = dropout.build_dropout_params(init_params,
                                                               dropout_init_key)

            # this pattern allows changing the energy parameters on-the-fly
            def energy_fn_template(energy_params):
                def energy_fn(pos, neighbor, **dynamic_kwargs):
                    gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                               species=species,
                                               **dynamic_kwargs)
                    prior_energy = prior_potential(prior_fns, pos, neighbor,
                                                   **dynamic_kwargs)
                    return gnn_energy + prior_energy

                return energy_fn

        else:
            # TODO - Initialize neighbor_capacity_multiple large enough to not cause overflows - this depends on first
            # TODO - structure passed.
            neighbor_capacity_multiple = 2.7    # Hard coded for ANI1-x dataset.
            config = neural_networks.initialize_nequip_cfg(n_species, r_cut)
            # TODO adjust A to nm in NequIP
            if species is None:
                species = jnp.zeros(init_pos.shape[0])
            atoms = nn.one_hot(species, n_species)
            # neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
            #     displacement, box, config, atoms=atoms, dr_threshold=dr_thresh,
            #     capacity_multiplier=neighbor_capacity_multiple,
            #     fractional_coordinates=True,
            #     disable_cell_list=True)
            neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
                displacement, box, config, atoms=None, dr_threshold=dr_thresh,
                capacity_multiplier=neighbor_capacity_multiple,
                fractional_coordinates=True,
                disable_cell_list=True)

            nbrs_init = neighbor_fn.allocate(init_pos,
                                             extra_capacity=extra_capacity)

            if isinstance(model_init_key, list):
                # ensemble of neural networks not needed together with dropout
                init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                       atoms=atoms, **energy_kwargs)
                               for key in model_init_key]
            else:
                init_params = init_fn(model_init_key, init_pos,
                                      neighbor=nbrs_init,
                                      atoms=atoms, **energy_kwargs)

            def energy_fn_template(energy_params):
                def energy_fn(pos, neighbor, **dynamic_kwargs):
                    # gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                    #                            atoms=atoms, **dynamic_kwargs)
                    _species = dynamic_kwargs.pop('species', None)
                    if _species is None:
                        raise ValueError('Species needs to be passed to energy_fn')
                    atoms_comp = nn.one_hot(_species, n_species)
                    gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                               atoms=atoms_comp, **dynamic_kwargs)
                    prior_energy = prior_potential(prior_fns, pos, neighbor,
                                                   **dynamic_kwargs)
                    return gnn_energy + prior_energy

                return energy_fn



    elif model == 'ANI-1x':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.7,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)  # create neighborlist for init of GNN

        # init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
        #     displacement, r_cut, n_species, init_pos, nbrs_init,
        #     kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        # )
        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist_ANI1x(
            displacement, r_cut, n_species, positions_test=None, neighbor_test=None,
            max_edges=max_edges, max_angles=max_angles, kbt_dependent=kbt_dependent, embed_size=64, init_kwargs=mlp_init
        )

        if isinstance(model_init_key, list):
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos, neighbor=nbrs_init,
                                  species=species, **energy_kwargs)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            # Took out species in partial() to force that species is passed, as this varies.
            gnn_energy = partial(GNN_energy, energy_params)
            def energy(R, neighbor, species, **dynamic_kwargs):
                return gnn_energy(positions=R, neighbor=neighbor, species=species, **dynamic_kwargs)
            return energy

    elif model == 'QM7x':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        # TODO - Find correct capacity multiplier based on first sample input -> implies max edges
        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.7,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)  # create neighborlist for init of GNN

        # init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
        #     displacement, r_cut, n_species, init_pos, nbrs_init,
        #     kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        # )
        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist_ANI1x(
            displacement, r_cut, n_species, positions_test=None, neighbor_test=None,
            max_edges=max_edges, max_angles=max_angles, kbt_dependent=kbt_dependent, embed_size=64, init_kwargs=mlp_init
        )

        if isinstance(model_init_key, list):
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos, neighbor=nbrs_init,
                                  species=species, **energy_kwargs)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            # Took out species in partial() to force that species is passed, as this varies.
            gnn_energy = partial(GNN_energy, energy_params)
            def energy(R, neighbor, species, **dynamic_kwargs):
                return gnn_energy(positions=R, neighbor=neighbor, species=species, **dynamic_kwargs)
            return energy

    elif model == 'QM7x_prior':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        # TODO - Find correct capacity multiplier based on first sample input -> implies max edges
        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.7,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)  # create neighborlist for init of GNN

        # init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
        #     displacement, r_cut, n_species, init_pos, nbrs_init,
        #     kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        # )
        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist_ANI1x(
            displacement, r_cut, n_species, positions_test=None, neighbor_test=None,
            max_edges=max_edges, max_angles=max_angles, kbt_dependent=kbt_dependent, embed_size=64, init_kwargs=mlp_init
        )

        if isinstance(model_init_key, list):
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos, neighbor=nbrs_init,
                                  species=species, **energy_kwargs)

        # Iterate over mol ids, intialize the amber force field for each molecule, and save in a dictionary
        atom_molecule_unique = onp.unique(onp.load("/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/QM7x_DB/atom_molecule_QM7x.npy"))

        # amber_init = {}
        # for mol in atom_molecule_unique[:10]:
        #     if mol != '2066':    # Hard coded, Anton mentioned that topolgy file creation for this one was not possible.
        #         amber_init[int(mol)] = au.build_amber_energy_fn("/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPfiles/mol_"+str(mol)+".prmtop")

        # for mol in mol_id_data[:10]:
        #     if mol != 2066:    # Hard coded, Anton mentioned that topolgy file creation for this one was not possible.
        #         amber_init[int(mol)] = au.build_amber_energy_fn("/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPfiles/mol_"+str(mol)+".prmtop")

        # amber_init = []  # [0] * int(max(mol_id_data))
        # for mol in mol_id_data[:10]:
        #     if mol != 2066:  # Hard coded, Anton mentioned that topolgy file creation for this one was not possible.
        #         amber_init.append(au.build_amber_energy_fn(
        #             "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPfiles/mol_" + str(
        #                 mol) + ".prmtop"))

        amber_init = []
        for mol_id in list(id_mapping_rev.keys()):
            amber_init.append(au.build_amber_energy_fn(
                "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPFiles/mol_" + str(
                    id_mapping_rev[mol_id]) + "_AC.prmtop"))



        # callback = lambda mol_id, R: amber_init[mol_id](R)
        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            # Took out species in partial() to force that species is passed, as this varies.
            gnn_energy = partial(GNN_energy, energy_params)

            def energy(R, neighbor, species, mol_id, **dynamic_kwargs):
                # amber_prior_energy = jax.pure_callback(callback, onp.array(0., dtype='float32'), (mol_id, R))
                # amber_prior_energy = amber_init[mol_id.astype(int)](R)
                # amber_prior_energy = amber_init[int(atom_molecule_unique[0])](R)
                # amber_prior_energy = amber_init[mol_id](R)
                idx = mol_id
                amber_prior_energy = lax.switch(idx, amber_init, R)
                return gnn_energy(positions=R, neighbor=neighbor, species=species, **dynamic_kwargs) + (amber_prior_energy / 10**4)

            return energy


    elif model == 'NequIP_QM7x_prior':
        r_cut = 5.0
        # r_cut = 6.0
        # r_cut = 4.0
        n_species = 100
        extra_capacity = 0
        dr_thresh = 0.05

        if prior_constants is not None:
            prior_fns = select_priors(displacement, prior_constants, prior_idxs,
                                      kbt)
            print('Using the following priors:')
            [print(key) for key in prior_fns]
        else:
            print('Using no priors')
            prior_fns = None

        # TODO - Initialize neighbor_capacity_multiple large enough to not cause overflows - this depends on first
        # TODO - structure passed.
        neighbor_capacity_multiple = 2.7    # Hard coded for ANI1-x dataset.
        # config = neural_networks.initialize_nequip_cfg(n_species, r_cut)
        config = neural_networks.initialize_nequip_cfg_MaxSetup(n_species, r_cut)
        if species is None:
            species = jnp.zeros(init_pos.shape[0])
        atoms = nn.one_hot(species, n_species)
        neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
            displacement, box, config, atoms=None, dr_threshold=dr_thresh,
            capacity_multiplier=neighbor_capacity_multiple,
            fractional_coordinates=True,
            disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos,
                                         extra_capacity=extra_capacity)

        if isinstance(model_init_key, list):
            # ensemble of neural networks not needed together with dropout
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   atoms=atoms, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos,
                                  neighbor=nbrs_init,
                                  atoms=atoms, **energy_kwargs)

        # amber_init = []
        # # prior_fns_init = []
        # for counter_map, mol_id in enumerate(list(id_mapping_rev.keys())):
        #     amber_fn = au_nequip.build_amber_energy_fn(
        #         "/home/sebastien/FreeEnergy_March/myjaxmd/examples/FreeEnergyScripts/QM7x Scripts/PRMTOPFiles/mol_" + str(
        #             id_mapping_rev[mol_id]) + "_AC.prmtop", unit_box_size=1000)
        #     amber_init.append(amber_fn)
        #     print("Initialized amber number: {}", counter_map)
        #     # prior_constants = {'LJ': (sigma, epsilon)}
        #     # prior_idxs = {'LJ': nonbond_pairs}
        #     # prior_fn = select_priors(displacement, prior_constants, prior_idxs, kbt=None)
        #     # prior_fns_init.append(prior_fn['LJ'])

        # TODO: Dividing amber_prior_energy by some factor needs more investigation
        def energy_fn_template(energy_params):
            def energy_fn(pos, neighbor, amber_energy, **dynamic_kwargs):
                # idx = mol_id
                # # jax.debug.print("Positions: {pos}", pos=pos)
                # amber_prior_energy = lax.switch(idx, amber_init, pos)
                # # prior_fn_executed = lax.switch(idx, prior_fns_init, (pos, neighbor))
                _species = dynamic_kwargs.pop('species', None)
                if _species is None:
                    raise ValueError('Species needs to be passed to energy_fn')
                atoms_comp = nn.one_hot(_species, n_species)
                # gnn_part  = gnn_energy_fn(energy_params, pos, neighbor,
                #                            atoms=atoms_comp, **dynamic_kwargs)
                # jax.debug.print("GNN energy: {gnn_part}", gnn_part=gnn_part)
                # jax.debug.print("Amber energy: {amber_prior_energy}", amber_prior_energy=amber_prior_energy)
                # gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                #                            atoms=atoms_comp, **dynamic_kwargs) + (amber_prior_energy) / 1000

                gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                           atoms=atoms_comp, **dynamic_kwargs) + amber_energy
                return gnn_energy

            return energy_fn

    elif model == 'NequIP_QM7x_priorInTarget':
        r_cut = 4.0
        n_species = 100
        extra_capacity = 0
        dr_thresh = 0.05

        if prior_constants is not None:
            prior_fns = select_priors(displacement, prior_constants, prior_idxs,
                                      kbt)
            print('Using the following priors:')
            [print(key) for key in prior_fns]
        else:
            print('Using no priors')
            prior_fns = None

        # TODO - Initialize neighbor_capacity_multiple large enough to not cause overflows - this depends on first
        # TODO - structure passed.
        neighbor_capacity_multiple = 2.7    # Hard coded for ANI1-x dataset.
        config = neural_networks.initialize_nequip_cfg_MaxSetup(n_species, r_cut)
        if species is None:
            species = jnp.zeros(init_pos.shape[0])
        atoms = nn.one_hot(species, n_species)
        neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
            displacement, box, config, atoms=None, dr_threshold=dr_thresh,
            capacity_multiplier=neighbor_capacity_multiple,
            fractional_coordinates=True,
            disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos,
                                         extra_capacity=extra_capacity)

        if isinstance(model_init_key, list):
            # ensemble of neural networks not needed together with dropout
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   atoms=atoms, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos,
                                  neighbor=nbrs_init,
                                  atoms=atoms, **energy_kwargs)


        def energy_fn_template(energy_params):
            def energy_fn(pos, neighbor, amber_energy, **dynamic_kwargs):
                _species = dynamic_kwargs.pop('species', None)
                if _species is None:
                    raise ValueError('Species needs to be passed to energy_fn')
                atoms_comp = nn.one_hot(_species, n_species)
                gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                           atoms=atoms_comp, **dynamic_kwargs)# + amber_energy
                return gnn_energy

            return energy_fn

    elif model == 'Nequip_HFE':
        ## NEQUIP MODEL WITH NO PRIORS

        ## MODEL PARAMETERS
        r_cut = 4.0
        n_species = 100
        extra_capacity = 0
        dr_thresh = 0.05
        neighbor_capacity_multiple = 2.7  # Hard coded for ANI1-x dataset.
        config = neural_networks.initialize_nequip_cfg_MaxSetup(n_species, r_cut)

        ## NEQUIP FUNCTIONS
        neighbor_fn, init_fn, gnn_energy_fn = energy.nequip_neighbor_list(
            displacement, box, config, atoms=None, dr_threshold=dr_thresh,
            capacity_multiplier=neighbor_capacity_multiple,
            fractional_coordinates=True,
            disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(init_pos,
                                         extra_capacity=extra_capacity)

        #LOAD UVAC PARAMETERS
        with open(vac_model_path, 'rb') as pickle_file:
            init_params = pickle.load(pickle_file)

        _one_hot = partial(one_hot, num_classes=n_species)

        ## CONVERSION & SCALING FACTORS
        std = jnp.float32(1.3413871350479776) #eV
        mean_energy = jnp.float32(-8705.874526590613) #eV
        eV_kJ = jnp.float32(96.4869)
        kJ_kcal = jnp.float32(0.239001)


        ## DEFINING ENERGY FUNCTION
        def energy_fn_template(energy_params):
            GNN_energy = partial(gnn_energy_fn, energy_params)
            def energy_fn(pos, neighbor, species, **dynamic_kwargs):
                atoms = _one_hot(species)
                U_GNN = GNN_energy(position=pos, neighbor=neighbor, species=species, atoms=atoms,
                                   **dynamic_kwargs)

                return (U_GNN*std + mean_energy)*eV_kJ*kJ_kcal  #scaling and converting eV to kcal/mol

            return energy_fn

    else:
        raise ValueError('The model' + model + 'is not implemented.')

    return energy_fn_template, neighbor_fn, init_params, nbrs_init


def initialize_nequip_cfg() -> ConfigDict:
    """Initialize configuration based on values of Paper. Similar to def test_config() in nequip_test.py"""
    config = ConfigDict()

    config.multihost = False

    config.seed = 0
    config.split_seed = 0
    config.shuffle_seed = 0

    config.epochs = 1000000
    config.epochs_per_eval = 1
    config.epochs_per_checkpoint = 10  # 1

    # keep checkpoint every n checkpoints, -1 only keeps last
    # works only on non-multihost, will error out on multi-host training
    # note that this works together with config.epochs_per_checkpoint, i.e. the
    # checkpointing will only be called on multiples of epochs_per_checkpoint
    config.keep_every_n_checkpoints = -1

    config.learning_rate = 1e-3 # TODO: Paper uses 1e-2
    config.schedule = 'constant'
    config.max_lr_plateau_epochs = 200  # TODO: Not sure what this means

    config.train_batch_size = 5  # 2
    config.test_batch_size = 5  # 2

    config.model_family = 'nequip'

    # network
    config.graph_net_steps = 5  #2  #(3-5 work best -> Paper) TODO: Not sure what is used in Paper - number of NequIP convolutional layers -> equivalent to number interaction layers num_layers
    config.nonlinearities = {'e': 'raw_swish', 'o': 'tanh'}
    config.use_sc = True
    config.n_elements = 8  # TODO - Number of chemical elements in the input data - Can this just be set larger?
    # config.hidden_irreps = '64x0o + 64x0e + 64x1o + 64x1e'  # '4x0e + 2x1e'    # If parity=True contains o, else only e
    # (irreducible representation of hidden/latent features) (Removing odd oftern returns same results -> Paper) (32
    # would be a good default. For accuracy increase, for performance decrease.)
    config.hidden_irreps = '64x0e + 64x1e'
    # config.sh_irreps = '1x0e + 1x1o'  # l=1 and parity=True, # '1x0e + 1x1e'    # Checked implementation in Pytorch (irreducible representations on the edges)
    config.sh_irreps = '1x0e + 1x1e'
    config.num_basis = 8    # (8 usually works best -> Paper)
    config.r_max = 4.0  # 2.5
    config.radial_net_nonlinearity = 'raw_swish'
    config.radial_net_n_hidden = 64  # 8  # Smaller is faster
    config.radial_net_n_layers = 3  # 2 #(1-3 work best -> Paper)

    # average number of neighbors per atom, used to divide activations are sum
    # in the nequip convolution, helpful for internal normalization.
    config.n_neighbors = 10.    # TODO: Do I need to precompute this or rough value ok?

    # Standard deviation used for the initializer of the weight matrix in the
    # radial scalar MLP
    config.scalar_mlp_std = 4.  # TODO - Just use default value?

    # config.train_dataset = ['harder_silicon']
    # config.test_dataset = ['harder_silicon']
    # config.validation_dataset = ['harder_silicon']

    config.pretraining_checkpoint = None
    # start from last or a specific pretraining checkpoint
    # options: 'last', or 'ckpt_number', where 'ckpt_number' is string, e.g. '10'
    # for multi-host training, can only be 'last'
    config.pretraining_checkpoint_to_start_from = 'last'    # TODO - Is it correct like this?

    # The loss is computed as three terms E + lam_F * F + lam_S * S where the
    # first term computes the MSE of the energy, the second computes the MSE of
    # the forces and the last term computes the MSE of the stress. The
    # `force_lambda` and `stress_lambda` parameters determine the relative
    # weighting of the terms.
    config.energy_lambda = 1.0
    config.force_lambda = 1000.0    # 1.0   # TODO: Find proper weighting, took value from Paper for MD-17
    config.stress_lambda = 0.0
    config.bandgap_lambda = 0.0

    # # TODO - Not sure about following values, just took given values.
    # config.energy_loss = ('huber', 0.01)
    # config.force_loss = ('huber', 0.01)
    # config.stress_loss = ('huber', 10.0)
    # config.bandgap_loss = 'L2'

    # 'norm_by_n', 'norm_by_3n', or 'unnormed', applies both to loss and
    # metrics computation
    config.force_loss_normalization = 'norm_by_3n'

    # If L2 regularization is used, then the optimizer is switched to AdamW.
    config.l2_regularization = 0.0

    config.optimizer = 'adam'

    # The epoch size controls the number of crystals in one epoch of data. If the
    # `epoch_size` is set to -1 then it is equal to the size of the dataset. This
    # is useful for large datasets where it is inconvenient to wait for a whole
    # pass through the training data to finish before outputting statistics. It
    # also allows datasets of different sizes to be compared on equal footing.
    config.epoch_size = -1
    config.eval_size = -1

    # By default, we do not restrict the number of atoms for the current system.
    config.max_atoms = -1

    # Scale and shift need to be set -> TODO: Just set 1 and zero? This should simply return true energy
    config.scale = 1.
    config.shift = 0.
    return config


def initialize_Nequip_model(species, displacement, box, init_pos):
    """Setup Nequip model. Similar to setuo() in class NequipTest of nequip_test.py"""
    config = initialize_nequip_cfg()

    key = random.PRNGKey(0)

    n = init_pos.shape[0]
    atoms = jnp.zeros((n, 8))
    for i in range(jnp.count_nonzero(species)):
        atoms = atoms.at[i, species[i]-1].set(1)
    # for j in range(jnp.count_nonzero(species), len(species)):
    #     atoms = atoms.at[j, 25].set(1)


    neighbor_fn, init_fn, energy_fn = energy.nequip_neighbor_list(
        displacement, box, config, atoms=atoms,
        fractional_coordinates=True,
        disable_cell_list=True)

    nbrs_init = neighbor_fn.allocate(init_pos)  # TODO: Might be necesarry to increase extra_capacity, default=1.25

    init_params = init_fn(key, init_pos, nbrs_init, atoms=atoms)
    def energy_fn_template(energy_params):
        energy_fn_init = partial(energy_fn, energy_params)

        def energy(pos, neighbor, species, **dynamic_kwargs):
            # Set up converting species to atoms
            # atoms = jnp.zeros((pos.shape[0], 94))
            # for i in range(jnp.count_nonzero(species)):
            #     atoms = atoms.at[i, species[i] - 1].set(1)
            # atoms = jax.nn.one_hot(species, 95)[:, 1:]
            atoms = jax.nn.one_hot(species, 9)[:, 1:]

            return energy_fn_init(pos, neighbor, atoms=atoms)

        return energy

    return energy_fn_template, neighbor_fn, init_params, nbrs_init



def initialize_simulation(init_class, model, target_dict=None, x_vals=None,
                          key_init=0, fractional=True, integrator='Nose_Hoover',
                          wrapped=True, kbt_dependent=False,
                          prior_constants=None, prior_idxs=None,
                          dropout_init_seed=None, vac_model_path=None, load_trajectories=False, loaded_params=None):
    key = random.PRNGKey(key_init)
    model_init_key, simulation_init_key = random.split(key, 2)

    box = init_class.box
    box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
    r_inits = init_class.r_init

    if fractional:
        r_inits = scale_fn(r_inits)

    multi_trajectory = r_inits.ndim > 2
    init_pos = r_inits[0] if multi_trajectory else r_inits

    displacement, shift = space.periodic_general(
        box_tensor, fractional_coordinates=fractional, wrapped=wrapped)

    energy_kwargs = {}
    if kbt_dependent:
        # to allow init of kbt_embedding
        energy_kwargs['kT'] = init_class.kbt

    energy_fn_template, neighbor_fn, init_params, nbrs = select_model(
        model, init_pos, displacement, box, model_init_key, init_class.kbt,
        init_class.species, x_vals, fractional, kbt_dependent,
        prior_idxs=prior_idxs, prior_constants=prior_constants,
        dropout_init_seed=dropout_init_seed, vac_model_path=vac_model_path, **energy_kwargs
    )

    energy_fn_init = energy_fn_template(init_params)

    # INITIALISING ENERGY FUNCTION WITH LOADED PARAMETERS
    if loaded_params is not None:
        energy_fn_init_loaded = energy_fn_template(loaded_params)

    # setup simulator
    if integrator == 'Nose_Hoover':
        simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     chain_length=3, chain_steps=1)
    elif integrator == 'Langevin':
        if model == 'Nequip_HFE':
            simulator_template = partial(simulate.nvt_langevin, shift_fn=shift,
                                         dt=init_class.dt, kT=init_class.kbt,
                                         gamma=1. / round(10 ** 3 / 48.8882129, 4))  ## CONVERTING TO PS^-1
        else:
            simulator_template = partial(simulate.nvt_langevin, shift=shift,
                                         dt=init_class.dt, kT=init_class.kbt,
                                         gamma=100.)
    elif integrator == 'NPT':
        chain_kwargs = {'chain_steps': 1}
        simulator_template = partial(simulate.npt_nose_hoover, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     pressure=init_class.ref_press,
                                     barostat_kwargs=chain_kwargs,
                                     thermostat_kwargs=chain_kwargs)
    elif integrator == 'NVE':
        simulator_template = partial(simulate.nve, shift_fn=shift,
                                     dt=init_class.dt)
    else:
        raise NotImplementedError('Integrator string not recognized!')

    init, _ = simulator_template(energy_fn_init)
    # init = jit(init)  # avoid throwing initialization NaN for debugging NaNs

    # INIT FUNCTION WITH LOADED PARAMETERS
    if loaded_params is not None:
        init_loaded, _ = simulator_template(energy_fn_init_loaded)

    if integrator == 'NVE':
        init = partial(init, kT=init_class.kbt)
    # box only used in NPT: needs to be box tensor as 1D box leads to error as
    # box is erroneously mapped over N dimensions (usually only for eps, sigma)


    ## DEFINING INIT SIM STATE FN
    ## THE ONLY DIFFERENCE FOR HFE CASE (FROM ORIGINAL SCRIPT) IS THE PASSING OF THE SPECIES
    if model == 'Nequip_HFE':
        # IF LOADING PARAMS, THEN INITIALISE BOTH VACUUM AND WATER STATES
        if loaded_params is not None:
            def init_sim_state(rng_key, pos):
                nbrs_update = nbrs.update(pos)

                state_vac = init(rng_key, pos, mass=init_class.masses, neighbor=nbrs_update,
                                 box=box_tensor, species=init_class.species, **energy_kwargs)
                state_wat = init_loaded(rng_key, pos, mass=init_class.masses, neighbor=nbrs_update,
                                        box=box_tensor, species=init_class.species, **energy_kwargs)
                return (state_vac, nbrs_update), (state_wat, nbrs_update)
        else:
            def init_sim_state(rng_key, pos):
                nbrs_update = nbrs.update(pos)

                state = init(rng_key, pos, mass=init_class.masses, neighbor=nbrs_update, box=box_tensor,
                             species=init_class.species, **energy_kwargs) # difference here is adding the species
                return state, nbrs_update

    else:
        ## THIS IS THE ORIGINAL CODE
        def init_sim_state(rng_key, pos):
            nbrs_update = nbrs.update(pos)
            state = init(rng_key, pos, mass=init_class.masses, neighbor=nbrs_update,
                         box=box_tensor, **energy_kwargs)
            return state, nbrs_update

 #    if multi_trajectory:
 #       n_inits = r_inits.shape[0]
 #       init_keys = random.split(simulation_init_key, n_inits)
 #       sim_state = vmap(init_sim_state)(init_keys, r_inits)
 #   else:
 #       sim_state = init_sim_state(simulation_init_key, init_pos)

    ## THE PRECOMPUTED TRAJECTORIES WERE SAVED WITHOUT NEIGHBOUR LISTS
    ## BUT THEY WERE SAVED WITH A STATE (sim_state = state, nbr_list)
    ## HERE, WHEN LOADING TRAJECTORIES, JUST PASS THE NBR FUNCTION TO BE COMBINED WITH THE STATE LATER ON
    ## THIS IS ALL DEPENDENT ON HOW YOU WANT TO LOAD THE TRAJECTORIES, BUT THIS WAY IS CERTAINLY QUICK
    if not load_trajectories:
        ## THIS IS THE ORIGINAL CODE
        if multi_trajectory:
            n_inits = r_inits.shape[0]
            init_keys = random.split(simulation_init_key, n_inits)
            sim_state = vmap(init_sim_state)(init_keys, r_inits)
        else:
            sim_state = init_sim_state(simulation_init_key, init_pos)
    else:
        #IF LOADING TRAJECTORIES, JUST PASS THE NBR FUNCTION TO BE COMBINED WITH THE STATE LATER ON
        sim_state = nbrs.update(init_pos)

    if target_dict is None:
        target_dict = {}
    compute_fns, targets = build_quantity_dict(
        init_pos, box_tensor, displacement, energy_fn_template, nbrs,
        target_dict, init_class)

    simulation_funs = (simulator_template, energy_fn_template, neighbor_fn)
    return sim_state, init_params, simulation_funs, compute_fns, targets
