try:
    import MDAnalysis
    from lammps import format
except ModuleNotFoundError:
    pass

from ase.io import read
from jax_md import space
import numpy as onp


def read_log(filename, key_dict):
    """Reads lammps log file of runs (each run is a dictionary with thermo
    fields as keys, storing the values over time) and creates arrays with all
    timesteps (ommiting the first). Option to specify if wanted quantity is an
    array or vector via key_dict.
    Args:
        filename: String providing the location of the log file.
        key_dict: Dictionary with keywords of wanted lammps quanitity and
                  value equal to size of array.

    Example of lammps log file:
        Step Temp c_stresstensor[1] c_stresstensor[2]
       0    302.71462     98962684     79064480
     500    307.37374     80883783     75343909
    1000    294.45111     65954646     90334534

    The key_dict would look like this:
        key_dict = {'Step': 1, 'Temp': 1, c_stresstensor': 2}

    Returns:
        Tuple of jnp arrays of position, force and length of the box.
    """
    log = format.LogFile(filename).runs  # info: thermo+thermostyle
    n_runs = len(log)  # Number of MD runs

    result_dict = {}
    for key in key_dict:
        size = key_dict[key]
        if size == 1:
            results = onp.asarray(log[n_runs - 1][key])
            results = onp.delete(results, 0, axis=0)  # removing timestep 0
            result_dict[key] = results
        else:
            n_steps = len(log[n_runs - 1][key + '[1]'])
            results = onp.zeros((n_steps, size))
            for i in range(1, size + 1):
                vector_key = key + f'[{i}]'
                results[:, i - 1] = onp.asarray(log[n_runs - 1][vector_key])
            results = onp.delete(results, 0, axis=0)  # removing timestep 0
            result_dict[key] = results

    return result_dict


def read_dump(filename):
    """Reads a lammps dumpfile (sorted!) and generates position and force arrays
    for each molecule at each timestep as (Nstep x Nmol x Natoms/molecule x 3)
    and the general box length. We need the masses of all Natoms as an
    array([m1,m2,...,mN]).T with size (Natoms/molecule x 1).
    Args:
        filename: String providing the location of the dump file.

    Returns:
        Tuple of jnp arrays of position, force and length of the box.
    """
    dump = read(filename, index=':', format='lammps-dump-text')
    n_steps = len(dump)  # Nsteps
    n_atoms = dump[0].numbers.shape[0]

    length = dump[0].cell.cellpar()[0] / 10  # convert to nm
    box = onp.array([length, length, length])
    _, shift = space.periodic(box)

    # not using the first timestep t=0
    positions = onp.zeros((n_steps - 1, n_atoms, 3))
    forces = onp.zeros((n_steps - 1, n_atoms, 3))

    for j in range(1, n_steps):
        positions[j - 1, :, :] = dump[j].positions / 10  # angström -> nm
        forces[j - 1, :, :] = dump[
                                  j].get_forces() * 41.84  # kcal/mol-A in kJ/mol-nm

    # shift all positions which are out of box
    positions = shift(positions, 0)

    return positions, forces, length


def trr_to_numpy(filegro, filetrr, force=False):
    """Load GROMACS trajectory trr file into numpy arrays
    and return positions and forces in nm. Option to ommit
    forces.
     Args:
        filegro: String providing the location of the GRO
                                file to load the structure.
        filetrr: String providing the location of the TRR
                                file to load the trajectory.
        force: If True, also outputs the force array.

    Returns:
        Tuple of jnp arrays of position and force or position.
    """
    u = MDAnalysis.Universe(filegro, filetrr)

    # Use this command if given trajectory with solvent to
    # select prefered atom groups (in this case: all heavy atoms)
    # atoms = u.select_atoms('not resname SOL and not (name H* or name CH* or type O)')
    atoms = u.atoms
    n_snaps = len(u.trajectory)
    n_atoms = atoms.positions.shape[0]

    positions = onp.zeros((n_snaps, n_atoms, 3))
    if force:
        forces = onp.zeros((n_snaps, n_atoms, 3))

    i = 0
    for ts in u.trajectory:
        positions[i] = atoms.positions / 10  # A -> nm
        if force:
            forces[i] = atoms.forces * 10  # kJ/mol*A -> kJ/mol*nm
        i += 1

    if force:
        return positions, forces
    else:
        return positions
