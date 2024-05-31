import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import numpy as onp
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_3d_coordinates(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    # Get atom positions
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conformer = mol.GetConformer()
    positions = [[conformer.GetAtomPosition(i).x,
                  conformer.GetAtomPosition(i).y,
                  conformer.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]

    species = []
    for i in atom_symbols:
        if i == 'C':
            species.append(6)
        elif i == 'H':
            species.append(1)
        elif i == 'O':
            species.append(8)
        elif i == 'N':
            species.append(7)
        elif i == 'Cl':
            species.append(17)

    species = onp.array(species)
    positions = onp.array(positions)
    return positions, species, mol


def pdb_to_3d_coordinates(pdb_path, add_H=False):
    mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)

    # Generate 3D coordinates
    if add_H:
        mol = Chem.AddHs(mol)
        # AllChem.UFFOptimizeMolecule(mol)
        # sys.exit("Currently doesnt work adding hydrogens properly. Positions are zero.")

    # AllChem.MMFFOptimizeMolecule(mol)

    # Get atom positions
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conformer = mol.GetConformer()
    positions = [[conformer.GetAtomPosition(i).x,
                  conformer.GetAtomPosition(i).y,
                  conformer.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]

    species = []
    for i in atom_symbols:
        if i == 'C':
            species.append(6)
        elif i == 'H':
            species.append(1)
        elif i == 'O':
            species.append(8)
        elif i == 'N':
            species.append(7)
        elif i == 'S':
            species.append(16)
        else:
            raise NotImplementedError
    species = onp.array(species)
    positions = onp.array(positions)
    return positions, species, mol


if __name__ == "__main__":
    # Example usage:
    alanine = False
    GProtein_rdkit = True
    GProtein_Bio = False

    if GProtein_rdkit:
        # G Protein
        pdb_path = "GProtein/1pgb_processed.pdb"
        positions, species, mol = pdb_to_3d_coordinates(pdb_path, add_H=False)

    elif alanine:
        # Alanine dipeptide
        smiles = "CC(C(=O)O)N"  # Replace with your SMILES string
        positions, species, _ = smiles_to_3d_coordinates(smiles)
        onp.save("AlanineDipeptide/atom_positions.npy", positions)
        onp.save("AlanineDipeptide/atom_numbers.npy", species)
    else:
        raise NotImplementedError

    print("Debug")
