import os

os.environ["CUDA_VISIBLE_DEVICES"] = str("")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Avoid error in jax 0.4.25
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from IPython.display import Image

# Create a molecule
mol = Chem.MolFromSmiles('CCO')

# Generate 3D coordinates
AllChem.EmbedMolecule(mol, randomSeed=42)

# Display the original molecule
img_original = Draw.MolToImage(mol)
img_original.show()
#
# # Change the position of the second atom (index 1) in the molecule
# conf = mol.GetConformer()
# conf.SetAtomPosition(1, (5.0, 5.0, 5.0))  # Set new coordinates for atom at index 1
#
# # Display the modified molecule
# img_modified = Draw.MolToImage(mol)
# img_modified.show()