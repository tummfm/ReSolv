import os
import sys

import jax

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
import rdkit
from rdkit.Geometry import Point3D


import smiles_preprocess
from chemtrain import  neural_networks
from chemtrain.jax_md_mod import custom_space
import chemtrain.copy_nequip_amber_utils_qm7x as au_nequip

# GENERAL NOTES: Learn a Nequip potential based onQM7x dataset

if __name__ == '__main__':
    # Choose setup
    model = "NequIP_QM7x_priorInTarget"
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
    # NequIP with Prior
    # avg_force_rms = 3.406141379571665
    # mean_energy = -8712.330850999595


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

    # # NequIP model with prior
    # date = "281123"
    # lr_decay = 1e-4
    # initial_lr = 1e-3
    # num_epochs = 4
    # id_num = "Nequip_QM7x_All_WithAmber_" + str(num_epochs) + "epochs_iL" + str(initial_lr) + "_lrdecay" + str(
    #     lr_decay) + "_scaledTargets_LargeTrainingSet_Cutoff4A"
    # save_name = date + "_QM7x_Nequip_" + id_num + "_Trainer_ANI1x_Energies" + str(
    #     num_epochs) + "epochs_mlp" + mlp + "_Shift" + shift_b + "_Scale" + scale_b + "_" + train_on + "_epoch_4"

    best_params_path = '/home/sebastien/FreeEnergy/examples/FreeEnergyScripts/savedTrainers/' + str(
        save_name) + '_Params.pkl'

    if FreeSolvSamples:
        smiles_list = ['COc1c(ccc(c1C(=O)O)Cl)Cl', 'c1c(c(c(c(c1Cl)Cl)Cl)Cl)c2c(cc(c(c2Cl)Cl)Cl)Cl', 'CCCCCCCl', 'C',
                       'C(C(Cl)(Cl)Cl)Cl', 'c1ccc2c(c1)C(=O)c3ccc(cc3C2=O)N', 'C=C', 'C(=C(Cl)Cl)(Cl)Cl',
                       'CCCN(CCC)C(=O)SCCC', 'c1ccc2c(c1)Oc3c(c(c(c(c3Cl)Cl)Cl)Cl)O2', 'c1ccc(c(c1)N)Cl',
                       'c1c(cc(c(c1Cl)Cl)Cl)Cl', 'c1ccc2c(c1)Oc3ccccc3O2', 'C([N+](=O)[O-])(Cl)(Cl)Cl',
                       'c1ccc2c(c1)Oc3ccc(cc3O2)Cl', 'c1cc2c(cc1Cl)Oc3cc(c(c(c3O2)Cl)Cl)Cl',
                       'c1c2c(cc(c1Cl)Cl)Oc3c(c(c(c(c3Cl)Cl)Cl)Cl)O2', 'CO', 'C(=C\\Cl)\\Cl', 'COc1cccc(c1O)OC',
                       'Cc1cccc(c1C)Nc2ccccc2C(=O)O', 'COc1c(c(c(c(c1Cl)C=O)Cl)OC)O', 'CCCc1ccc(c(c1)OC)O',
                       'CN(C)C(=O)c1ccc(cc1)[N+](=O)[O-]', 'c1cc(c(c(c1c2cc(c(c(c2Cl)Cl)Cl)Cl)Cl)Cl)Cl', 'COC',
                       'c1cc(c(cc1c2cc(c(c(c2Cl)Cl)Cl)Cl)Cl)Cl', 'COC(CCl)(OC)OC', 'C(=C(Cl)Cl)Cl', 'CC(Cl)(Cl)Cl',
                       'c1cc2c(cc1Cl)Oc3ccc(cc3O2)Cl', 'C=C(Cl)Cl', 'c1ccc2c(c1)ccc3c2cccc3', 'N', 'COC(c1ccccc1)(OC)OC',
                       'CNC(=O)Oc1cccc2c1cccc2', 'c1ccc(cc1)S', 'c1cc(ccc1Cl)Cl', 'CCOC(OCC)Oc1ccccc1',
                       'CN(C)CCC=C1c2ccccc2CCc3c1cccc3', 'c1cc(c(c(c1)Cl)C#N)Cl', 'Cc1ccc(c2c1cccc2)C', 'C[N+](=O)[O-]',
                       'CC=O', 'CC(=O)Oc1ccccc1C(=O)O', 'CCCCCCCC(=O)OC', 'c1ccc2c(c1)Oc3cc(c(cc3O2)Cl)Cl',
                       'c1cc(cc(c1)Cl)Cl', 'C(C(Cl)Cl)Cl', 'c1(c(c(c(c(c1Cl)Cl)Cl)Cl)Cl)Cl', 'CC#C',
                       'c1c(cc(c(c1Cl)Cl)Cl)c2cc(c(c(c2Cl)Cl)Cl)Cl', 'CCCOC(=O)c1ccc(cc1)O', 'c1ccc(c(c1)O)Cl',
                       'CCCCOC(=O)c1ccc(cc1)O', 'CCNc1nc(nc(n1)Cl)NCC', 'c1(c(c(c(c(c1Cl)Cl)Cl)Cl)Cl)N(=O)=O', 'CCCCCCCCCC',
                       'Cc1cc2ccccc2cc1C', 'COc1ccc(cc1)C(=O)OC', 'c1(c(c(c(c(c1Cl)Cl)Cl)Cl)Cl)c2c(c(c(c(c2Cl)Cl)Cl)Cl)Cl',
                       'C(CCl)CCl', 'CC#N', 'c1ccc2cc(ccc2c1)N', 'CS', 'c1cc(c(cc1c2c(c(cc(c2Cl)Cl)Cl)Cl)Cl)Cl',
                       'c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N', 'CN(C)C(=O)c1ccccc1', 'c1ccc(cc1)c2ccccc2',
                       'Cn1cnc2c1c(=O)n(c(=O)n2C)C', 'c1cc(c(cc1Cl)c2cc(c(c(c2)Cl)Cl)Cl)Cl', 'c1cc(ccc1N)Cl',
                       'c1cc(c(cc1O)Cl)Cl', 'c1ccc2cc(ccc2c1)O', 'c12c(c(c(c(c1Cl)Cl)Cl)Cl)Oc3c(c(c(c(c3Cl)Cl)Cl)Cl)O2',
                       'C(CCl)OCCCl', 'CCCCCCCCC=O', 'CC(Cl)Cl', 'CC1(Cc2cccc(c2O1)OC(=O)NC)C', 'Cc1ccc(cc1)C(=O)N(C)C',
                       'c1cc(c(c(c1)Cl)Cl)Cl', 'CCCCCCl', 'c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)O)N', 'CCNc1nc(nc(n1)SC)NC(C)C',
                       'c1cc(c(c(c1Cl)Cl)Cl)Cl', 'c1cc(c(c(c1)Cl)c2c(cccc2Cl)Cl)Cl', 'c1ccc2c(c1)C(=O)c3cccc(c3C2=O)N',
                       'C(C(Cl)Cl)(Cl)Cl', 'c1c2c(cc(c1Cl)Cl)Oc3cc(c(cc3O2)Cl)Cl', 'CCSC', 'CN(C)CCOC(c1ccccc1)c2ccccc2',
                       'c1c(c(cc(c1Cl)Cl)Cl)Cl', 'Cc1ccc2cc(ccc2c1)C', 'CSSC', 'CCCCN(CC)C(=O)SCCC', 'c1ccc(cc1)Cn2ccnc2',
                       'Cc1cccc2c1cccc2', 'C(C(Cl)(Cl)Cl)(Cl)Cl', 'C=CCl', 'c1ccc(cc1)c2c(cc(cc2Cl)Cl)Cl', 'CC=C',
                       'COC(CC#N)(OC)OC', 'CCCCCCCCCC(=O)C', 'CCc1cccc2c1cccc2', 'c1ccc(cc1)c2cc(ccc2Cl)Cl',
                       'COC(=O)c1ccc(cc1)[N+](=O)[O-]', 'COc1ccccc1OC', 'CSc1ccccc1', 'c1ccc2c(c1)cccn2', 'C(Cl)Cl',
                       'CCc1cccc(c1N(COC)C(=O)CCl)CC', 'S']
    elif use_alanine_dipeptide:
        smiles_list = ["CC(C(=O)NC)NC(=O)C"]

    for count, smile in enumerate(smiles_list):
        if use_FreeSolv:
            if use_only_equilibrium:
                # Load data - already shuffled
                pad_pos = onp.load("QM7x_DB/equilibrium_shuffled_atom_positions_QM7x.npy")[index]
                pad_species = onp.load("QM7x_DB/equilibrium_shuffled_atom_numbers_QM7x.npy")[index]
            else:
                pad_pos = onp.load("QM7x_DB/shuffled_atom_positions_QM7x.npy")[index]
                pad_species = onp.load("QM7x_DB/shuffled_atom_numbers_QM7x.npy")[index]
        elif use_Alanine:
            wandb.init(
                project='Alanine_simulation_Uvac',
                config={
                    "Time [ps]": str(total_time),
                    "date": "201123",
                }
            )

            smiles = "CC(C(=O)O)N"
            pad_pos = onp.load("AlanineDipeptide/atom_positions.npy")
            pad_species = onp.load("AlanineDipeptide/atom_numbers.npy")

            _, _, mol = smiles_preprocess.smiles_to_3d_coordinates(smiles)
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                x, y, z = pad_pos[i]
                conf.SetAtomPosition(i, Point3D(x, y, z))
            wandb.log({"Initial molecule": wandb.Molecule.from_rdkit(mol)})
        elif use_alanine_dipeptide:
            wandb.init(
                project='AlanineDipeptide_simulation_Uvac',
                config={
                    "Time [ps]": str(total_time),
                    "date": "161223",
                }
            )
            # Data from https://pubchem.ncbi.nlm.nih.gov/compound/N-Acetyl-L-alanine-methylamide#section=InChIKey
            # smiles = "CC(C(=O)NC)NC(=O)C"  # canonical smiles
            # smiles = "C[C@@H](C(=O)NC)NC(=O)C"  # isomeric smiles
            smiles = smile
            pad_pos, pad_species, mol = smiles_preprocess.smiles_to_3d_coordinates(smiles)
            # pad_pos_comp1, pad_species_comp1, mol_comp1 = smiles_preprocess.smiles_to_3d_coordinates("C[C@@H](C(=O)NC)NC(=O)C")


        elif use_GProtein:
            wandb.init(
                project='GProtein_simulation_Uvac_with_prior',
                config={
                    "Time [ps]": str(total_time),
                    "date": "121223",
                }
            )
            pdb_path = "GProtein/1pgb_processed.pdb"
            pad_pos, pad_species, mol = smiles_preprocess.pdb_to_3d_coordinates(pdb_path, add_H=False)
            conf = mol.GetConformer()
            # wandb.log({"Initial molecule": wandb.Molecule.from_rdkit(mol)})
        elif use_GProtein_Amber:
            # wandb.init(
            #     project='GProtein_simulation_Uvac_with_prior',
            #     config={
            #         "Time [ps]": str(total_time),
            #         "date": "131223",
            #     }
            # )
            prmtop_path = "GProtein/1pgb_protein.prmtop"
            pdb_path = "GProtein/1pgb_protein_tleap.pdb"
            import MDAnalysis as mda
            # u = mda.Universe(pdb_path)
            # pad_pos = onp.array(u.atoms.positions)
            pad_pos, pad_species, mol = smiles_preprocess.pdb_to_3d_coordinates(pdb_path, add_H=False)
        elif FreeSolvSamples:
            wandb.init(
                project='RunUvacNoPrior_FreeSolvSamplesThatFailedForAnton',
                config={
                    "Time [ps]": str(total_time),
                    "date": "161223",
                    "smiles": smile
                }
            )
            smiles = smile
            pad_pos, pad_species, mol = smiles_preprocess.smiles_to_3d_coordinates(smiles)
            print("Debug")

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

        gamma = 5 / time_conversion
        simulator_template = partial(simulate.nvt_langevin, shift_fn=shift_fn, dt=dt, kT=kbT,
                                     gamma=4.)

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

        # nbrs = nbrs_init
        init, update = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kbT)
        # update = jit(update)
        key, sub_key = random.split(key)
        nbrs = neighbor_fn(R_init)
        state = init(sub_key, R_init, mass=mass, neighbor=nbrs, species=pad_species)

        all_pos = []
        for _ in range(steps_equil + steps_prod):
            state, nbrs = new_update(state, nbrs, pad_species)
            R = state.position

            if FreeSolvSamples:
                sample_freq = 500
            elif use_alanine_dipeptide:
                sample_freq = 100
            else:
                sys.exit("No sample frequency specified")

            if (_ > steps_equil) and (_ % sample_freq == 0):
                energy, force = compute_energy_force(R, nbrs, pad_species)
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

        if FreeSolvSamples:
            onp.save("FreeSolvesSamplesThatFailedForAnton/" + name + "positions.npy", positions)
            onp.save("FreeSolvesSamplesThatFailedForAnton/" + name + "forces.npy", forces)
            onp.save("FreeSolvesSamplesThatFailedForAnton/" + name + "energies.npy", energies)
            onp.save("FreeSolvesSamplesThatFailedForAnton/" + name + "initial_position.npy", pad_pos)
            onp.save("FreeSolvesSamplesThatFailedForAnton/" + name + "initial_species.npy", pad_species)
        elif use_alanine_dipeptide:
            onp.save("AlanineDipeptide/" + name + "positions.npy", positions)
            onp.save("AlanineDipeptide/" + name + "forces.npy", forces)
            onp.save("AlanineDipeptide/" + name + "energies.npy", energies)
            onp.save("AlanineDipeptide/" + name + "initial_position.npy", pad_pos)
            onp.save("AlanineDipeptide/" + name + "initial_species.npy", pad_species)

        # # Save positions
        # onp.save("GProtein/sim_data/positions_Time"+str(total_time)+"ps.npy", positions)

        if use_Alanine:
            # Save molecule
            for i in range(mol.GetNumAtoms()):
                x, y, z = pad_pos[i]
                conf.SetAtomPosition(i, Point3D(x, y, z))
            wandb.log({"Final molecule": wandb.Molecule.from_rdkit(mol)})



energy_fn = energy_fn_template(loaded_trainer.best_params)


def return_forces(pos, box, species):
    energy_fn = energy_fn_template(loaded_trainer.best_params)
    _, neg_forces = jax.value_and_grad(energy_fn)(pos, box, species)
    pred_forces = - neg_forces
    return pred_forces