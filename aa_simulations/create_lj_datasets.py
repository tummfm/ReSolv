"""Helper file to re-structure LAMMPS dump into hdf5 dataset."""
import pathlib

import h5py
import numpy as onp
from lammps_util import read_dump

folder = 'data/LJ_datasets/'


def process_trajectory(simulation_str):
    masses = onp.array(1.)

    pathlib.Path(folder + simulation_str).mkdir(parents=True, exist_ok=True)

    dumpfile = folder + 'raw/' + simulation_str + '.trj'
    # logfile = folder + 'raw/' + simulation_str + '.log'

    positions, forces, length = read_dump(dumpfile, masses, False)

    onp.save(folder + simulation_str + '/positions', positions)
    onp.save(folder + simulation_str + '/forces', forces)
    onp.save(folder + simulation_str + '/length', length)
    print('Finsished creating forces, positions')

    # Pv, Sv = read_log(logfile)
    # Pv = read_log(logfile)

    # np.save(f"confs/virial_pressure_{saveas}", Pv)
    # np.save(f"confs/virial_stress_{saveas}", Sv)
    # print("Finsished creating virial pressure, stress")


    # -------------------------------------------------------------
    # Analysis
    print('Position 1=', positions[0, 1, :])
    # print("Forces 1=",forces[0,1,:])
    print('Position size=', positions.shape)
    print('Forces size=', forces.shape)
    # print("Virial pressure size=",Pv.shape)
    # print("Virial stress size=",Sv.shape)
    print('Length=', length)
    # print(positions)
    # print("Reduced = shape:", pos.shape)


def build_hdf5(simulations, n_atoms=1000):
    with h5py.File(folder + 'LJ_datasets.h5', 'w') as h5:
        for i, (simulation_str, temperature) in enumerate(simulations.items()):
            positions = onp.load(folder + simulation_str + '/positions.npy')
            forces = onp.load(folder + simulation_str + '/forces.npy')
            side_length = onp.load(folder + simulation_str + '/length.npy')
            density = n_atoms / side_length**3
            box = onp.eye(3) * side_length

            group = h5.create_group(f'Dataset{i}')
            group.create_dataset('positions', data=positions)
            group.create_dataset('forces', data=forces)
            group.create_dataset('box', data=box)
            group.create_dataset('temperature', data=temperature)
            group.create_dataset('density', data=density)


if __name__ == '__main__':
    identifyer = 'n_1000_t_1.5_l_8'
    # process_trajectory(identifyer)

    statepoints = {
        'n_1000_t_1_l_10': 1.,
        'n_1000_t_1_l_8': 1.,
        'n_1000_t_1.5_l_8': 1.5,
        'n_1000_t_1.5_l_10': 1.5
    }
    build_hdf5(statepoints)

