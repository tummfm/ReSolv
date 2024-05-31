import os
import sys

visible_device = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)


import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from jax_md import space

from chemtrain.jax_md_mod import custom_space, custom_quantity




def plot_1d_dihedral(ax, angles, labels, bins=60, degrees=True,
                     xlabel='$\phi$ in deg'):
    """Plot  1D histogram splines for a dihedral angle. """
    color = ['k', '#00A087FF', '#3C5488FF']
    line = ['--', '-', '-']

    n_models = len(angles)
    for i in range(n_models):
        if degrees:
            angles_conv = angles[i]
            hist_range = [-180, 180]
        else:
            angles_conv = onp.rad2deg(angles[i])
            hist_range = [-onp.pi, onp.pi]

        # Compute the histogram
        hist, x_bins = jnp.histogram(angles_conv, bins=bins, density=True, range=hist_range)
        width = x_bins[1] - x_bins[0]
        bin_center = x_bins + width / 2

        ax.plot(
            bin_center[:-1], hist, label=labels[i], color=color[i],
            linestyle=line[i], linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')

    return ax


def plot_histogram_free_energy(ax, phi, psi, kbt, degrees=True, title=""):
    """Plot 2D free energy histogram for alanine from the dihedral angles."""
    cmap = plt.get_cmap('magma')

    if degrees:
        phi = jnp.deg2rad(phi)
        psi = jnp.deg2rad(psi)

    h, x_edges, y_edges = jnp.histogram2d(phi, psi, bins=60, density=True)

    # h = jnp.log(h) * -(kbt / 4.184)  # kJ to kcal
    h = jnp.log(h) * (-kbt)

    x, y = onp.meshgrid(x_edges, y_edges)

    cax = ax.pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    ax.set_xlabel('$\phi$ in rad')
    ax.set_ylabel('$\psi$ in rad')
    ax.set_title(title)

    return ax, cax


def postprocess_fn(positions):
    # Compute the dihedral angles
    dihedral_idxs = jnp.array([[1, 3, 4, 6], [3, 4, 6, 8]])  # 0: phi    1: psi
    batched_dihedrals = jax.vmap(
        custom_quantity.dihedral_displacement, (0, None, None)
    )

    dihedral_angles = batched_dihedrals(positions, displacement_fn, dihedral_idxs)

    return dihedral_angles.T

if __name__ == '__main__':
    system_temperature = 298.15  # Kelvin
    boltzmann_constant = 0.00198720426  # in kcal / mol K
    kT = system_temperature * boltzmann_constant

    box = jnp.eye(3) * 1000
    displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)

    # TODO - Load U_vac_positions and U_wat_positions.
    # TODO - If not fractional, make it fractional
    # scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
    # U_vac_positions = lax.map(scale_fn, U_vac_positions)
    # U_vac_positions = ...
    # U_wat_positions = ...

    # ATTENTIon - be aware of the units
    # kJ_to_kcal = 0.2390057  # kJ -> kcal
    # eV_to_kcal_per_mol = 23.0605419  # eV -> kcal/mol

    # Use fractional coordinates
    U_vac_phi, U_vac_psi = postprocess_fn(U_vac_positions)
    U_wat_phi, U_wat_psi = postprocess_fn(U_wat_positions)

    labels = ["U_vac", "U_wat"]

    fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained", figsize=(9, 4))
    ax1 = plot_1d_dihedral(ax1, [U_vac_phi, U_wat_phi], labels, xlabel="$\phi\ [deg]$")
    ax2, cax = plot_1d_dihedral(ax2, [U_vac_psi, U_wat_psi], labels, xlabel="$\psi\ [deg]$")
    fig.legend(labels, ncols=len(labels), bbox_to_anchor=(0.5, 1.01), loc="lower center")

    fig.savefig("AlanineDipeptide/alanine_dipeptide_1D_dihedral_angles.pdf")

    labels = ["AA Reference"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 2, layout="constrained", figsize=(9, 3))
    ax1, _ = plot_histogram_free_energy(ax1, U_vac_phi, U_vac_psi, kT, title="U_vac")
    ax2, _ = plot_histogram_free_energy(ax2, U_wat_phi, U_wat_psi, kT, title="U_wat")

    cbar = fig.colorbar(cax)
    cbar.set_label('Free Energy (kcal/mol) - TODO units correct?')

    fig.savefig("AlanineDipeptide/alanine_dipeptide_free_energy_dihedral_angles.pdf")