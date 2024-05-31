import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Avoid error in jax 0.4.25
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

from jax import config, nn, jit
import pickle
config.update("jax_enable_x64", True)
import numpy as onp
from jax_md import space, partition, energy as energy_jaxmd, simulate, quantity
from jax import random, numpy as jnp, lax
from functools import partial
import wandb
import pandas as pd
import rdkit
from rdkit.Geometry import Point3D
import jax


import smiles_preprocess
from chemtrain import  neural_networks
from chemtrain.jax_md_mod import custom_space
import chemtrain.copy_nequip_amber_utils_qm7x as au_nequip

# GENERAL NOTES: Learn a Nequip potential based onQM7x dataset

if __name__ == '__main__':
    # Choose setup
    model = "NequIP_HFE"
    model_type = 'U_vac'
    use_FreeSolv = False
    use_only_equilibrium = False
    use_Alanine = False
    use_GProtein = False
    use_GProtein_Amber = False
    FreeSolvSamples = False
    use_alanine_dipeptide = True   # C6H12N2O2 -> correct?
    index = 0

    # NequIP without Prior
    avg_force_rms = 1.3413871350479776
    mean_energy = -8705.874526590613

    # Set up simulation
    time_conversion = round(10 ** 3 / 48.8882129, 7)  # 1 unit --> 1
    # ps
    kJ_to_kcal = 0.2390057  # kJ -> kcal
    eV_to_kcal_per_mol = 23.0605419  # eV -> kcal/mol
    k_b = 0.0083145107  # in kJ / mol K
    T = 300  # in K
    kbT = k_b * T  # in kJ / mol
    dt = 1e-3  # in ps
    total_time_in_ps = 100000
    total_time = total_time_in_ps  # in ps   -> we sample every 0.1ps, this results in 3200 samples. Added 1ps which is later cutoff.
    t_equil = 0  # in ps

    #Convert to simulation units
    dt *= time_conversion
    total_time *= time_conversion
    t_equil *= time_conversion
    kbT *= kJ_to_kcal

    steps_equil = int(t_equil / dt)
    steps_prod = int((total_time - t_equil) / dt)

    # Load trainer
    shift_b = "False"
    scale_b = "False"
    mlp = "4"
    train_on = "EnergiesAndForces"
    all_size = onp.load("QM7x_DB/shuffled_atom_energies_QM7x.npy").shape[0]
    num_epochs = 8
    date = "261023"
    id_num = "Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet"
    save_name = date + "_QM7x_Nequip_" + id_num + "_Trainer_ANI1x_Energies" + str(all_size) + "samples_" + str(
        num_epochs) + "epochs_mlp" + mlp + "_Shift" + shift_b + "_Scale" + scale_b + "_" + train_on + "_epoch_8"


    # best_params_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/' + str(
    #     save_name) + '_Params.pkl'

    if model_type == 'U_vac':
        best_params_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/261023_QM7x_Nequip_' \
                         'Nequip_QM7x_All_8epochs_iL5emin3_lrdecay1emin3_scaledTargets_LargeTrainingSet_Trainer_' \
                         'ANI1x_Energies4195237samples_8epochs_mlp4_ShiftFalse_ScaleFalse_EnergiesAndForces_epoch_8_' \
                         'Params.pkl'
    elif model_type == 'U_wat':
        t_prod_U_vac = 250
        t_equil_U_vac = 50
        initial_lr = 0.000001
        lr_decay = 0.1
        passed_seed = 7
        save_checkpoint = (date + "_t_prod_" + str(t_prod_U_vac) + "ps_t_equil_" +
                           str(t_equil_U_vac) + "ps_iL" + str(initial_lr) + "_lrd" +
                           str(lr_decay) + "_epochs" + str(num_epochs)
                           + "_seed" + str(passed_seed) + "_train_389" +
                           "mem_0.97")
        best_params_path = 'checkpoints/' + save_checkpoint + '_epoch499.pkl'

    if use_alanine_dipeptide:
        # wandb.init(
        #     project='AlanineDipeptide_simulation_Uvac',
        #     config={
        #         "Time [ps]": str(total_time),
        #         "date": "161223",
        #     }
        # )
        # Data from https://pubchem.ncbi.nlm.nih.gov/compound/N-Acetyl-L-alanine-methylamide#section=InChIKey
        # smiles = "CC(C(=O)NC)NC(=O)C"  # canonical smiles
        # smiles = "C[C@@H](C(=O)NC)NC(=O)C"  # isomeric smiles
        # pad_pos, pad_species, mol = smiles_preprocess.smiles_to_3d_coordinates(smiles)
        # pad_pos_comp1, pad_species_comp1, mol_comp1 = smiles_preprocess.smiles_to_3d_coordinates("C[C@@H](C(=O)NC)NC(=O)C")

        pd_alanine = pd.read_csv('../../../aa_simulations/alanine_dipeptide/C5.pdb', nrows=22,
                                 delim_whitespace=True, header=None)
        species_list = pd_alanine[2].to_list()
        pad_pos = pd_alanine.iloc[:, [5, 6, 7]].to_numpy()


        pad_species = []
        for atom in species_list:
            if atom in ('1HH3', '2HH3', '3HH3', 'H', 'HA', 'HB1', 'HB2', 'HB3'):
                pad_species.append(1)
            elif atom in ('CH3', 'C', 'CA', 'CB', 'CH3'):
                pad_species.append(6)
            elif atom in ('N'):
                pad_species.append(7)
            elif atom in ('O'):
                pad_species.append(8)
            else:
                raise NotImplementedError

        pad_species = onp.array(pad_species)
        # pad_pos = jnp.array(pad_pos)

    # Truncate padding
    num_atoms = onp.count_nonzero(pad_species)
    pad_pos = pad_pos[:num_atoms]
    pad_species = pad_species[:num_atoms]
    mass = []
    for spec in pad_species:
        if spec == 1:
            mass_add = 1.00784
        elif spec == 6:
            mass_add = 12.011
        elif spec == 7:
            mass_add = 14.007
        elif spec == 8:
            mass_add = 15.999
        elif spec == 16:
            mass_add = 32.06
        elif spec == 17:
            mass_add = 35.45
        mass.append(mass_add)
    mass = jnp.array(mass)

    pad_pos += onp.array([150., 150., 150.])
    box = jnp.eye(3) * 300

    key = random.PRNGKey(0)

    # Use jax.numpy
    box = jnp.array(box)
    R_init = jnp.array(pad_pos)

    # already scaled
    _, scale_fn = custom_space.init_fractional_coordinates(box)
    R_init = scale_fn(R_init)

    # # Setup LJ simulation
    # displacement_fn, shift_fn = space.periodic_general(box)

    # Load energy_fn to make predictions
    n_species = 100
    r_cut = 4.0

    dr_thresh = 0.05
    neighbor_capacity_multiple = 2.7  # Hard coded for ANI1-x dataset.
    displacement, shift_fn = space.periodic_general(box=box, fractional_coordinates=True)
    init_species = pad_species

    # Amber prior
    # amber_fn = au_nequip.build_amber_energy_fn(prmtop_path, displacement)

    # TODO : ATTENTION Be aware that config must be the same as with training!
    config = neural_networks.initialize_nequip_cfg_MaxSetup(n_species, r_cut)
    atoms = nn.one_hot(init_species, n_species)
    neighbor_fn, init_fn, gnn_energy_fn = energy_jaxmd.nequip_neighbor_list(
        displacement, box, config, atoms=None, dr_threshold=dr_thresh,
        capacity_multiplier=neighbor_capacity_multiple,
        fractional_coordinates=True,
        disable_cell_list=True)

    # # Use GNN for energy prediction
    # neighbor_fn = partition.neighbor_list(displacement_fn, box, r_cut,
    #                                       dr_threshold=0.05,
    #                                       capacity_multiplier=2.5,
    #                                       disable_cell_list=True,
    #                                       fractional_coordinates=True)
    nbrs_init = neighbor_fn(R_init)

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, **dynamic_kwargs):
            _species = dynamic_kwargs.pop('species', None)
            if _species is None:
                raise ValueError('Species needs to be passed to energy_fn')
            atoms_comp = nn.one_hot(_species, n_species)
            gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                       atoms=atoms_comp, **dynamic_kwargs)
            # convert to kcal/mol
            # amber_energy = 0 # amber_fn(pos)
            gnn_energy = (gnn_energy * avg_force_rms + mean_energy) * eV_to_kcal_per_mol
            return gnn_energy

        return energy_fn


    with open(best_params_path, 'rb') as pickle_file:
        loaded_params = pickle.load(pickle_file)

    energy_fn = energy_fn_template(loaded_params)

    gamma = 1. / round(10 ** 3 / 48.8882129, 4)
    simulator_template = partial(simulate.nvt_langevin, shift_fn=shift_fn, dt=dt, kT=kbT,
                                 gamma=gamma)

    forces = []
    energies = []
    positions = []


    @jit
    def new_update(state, nbrs, species):
        state = update(state, neighbor=nbrs, species=species)
        nbrs = nbrs.update(state.position)
        return state, nbrs

    @jit
    def compute_energy_force(R, nbrs, species):
        energy = energy_fn(R, neighbor=nbrs, species=species)
        force = quantity.force(energy_fn)(R, neighbor=nbrs, species=species)
        return energy, force

    # init, update = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kbT)
    init, update = simulate.nvt_langevin(energy_fn, shift_fn, dt, kbT, gamma)
    # update = jit(update)

    # key, sub_key = random.split(key, 101)
    nbrs = neighbor_fn(R_init)
    pad_species = jnp.array(pad_species)

    def init_fn(keys, R_init, mass, nbrs, pad_species)
        init(keys, R_init, mass=mass, neighbor)

    keys = random.split(key, 100)
    partial(init, mass=mass, neighbor=nbrs, species=pad_species)
    state_vmap = jax.vmap(init, in_axes=(0, None))(keys, R_init)

    # state = init(sub_key, R_init, mass=mass, neighbor=nbrs, species=pad_species)

    all_pos = []
    nbrs_vmap = jnp.tile(nbrs, 100)
    for _ in range(steps_equil + steps_prod):
        # state, nbrs = new_update(state, nbrs, pad_species)
        state_vmap, nbrs_vmap = jax.vmap(new_update, in_axes=(0, 0, None))(state_vmap, nbrs_vmap, pad_species)
        # R = state.position

        if use_alanine_dipeptide:
            sample_freq = 100
        else:
            sys.exit("No sample frequency specified")

        if (_ > steps_equil) and (_ % sample_freq == 0):
            energies, forces = jax.vmap(compute_energy_force, in_axes=(0, 0, None))(state_vmap, nbrs_vmap, pad_species)
            # energy, force = compute_energy_force(R, nbrs, pad_species)
            energies.append(energy)
            forces.append(force)
            positions.append(R)
            print(f'Step {_}')
            print(f'E = {energy}')
            print(f'F = {force}')

            if FreeSolvSamples or use_alanine_dipeptide:
                wandb.log({"Energy": energy})

    # Convert unit positions to real positions
    positions = onp.array([space.transform(box, pos) for pos in positions])

    # Save
    name = str(count) + '_SMILE' + smiles + '_' + str(total_time_in_ps) + 'ps_'

    if use_alanine_dipeptide:
        onp.save("AlanineDipeptide/" + name + "positions.npy", positions)
        onp.save("AlanineDipeptide/" + name + "forces.npy", forces)
        onp.save("AlanineDipeptide/" + name + "energies.npy", energies)
        onp.save("AlanineDipeptide/" + name + "initial_position.npy", pad_pos)
        onp.save("AlanineDipeptide/" + name + "initial_species.npy", pad_species)


