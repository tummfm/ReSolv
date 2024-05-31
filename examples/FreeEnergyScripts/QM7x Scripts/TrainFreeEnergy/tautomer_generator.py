import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from rdkit import Chem

# Define the SMIRKS notation
smirks = '[O:1]=[C:2]1[NH:3][CH2:5][CH:6]=[CH:4]1>>[O:1]=[C:2]1[NH:3][CH:5]=[CH:6][CH2:4]1'

# Split the SMIRKS notation into reactant and product parts
reactant_smarts, product_smarts = smirks.split(">>")

# Convert SMIRKS into SMARTS
# reactant_smarts = reactant_smarts.replace("[", "(").replace("]", ")")
# product_smarts = product_smarts.replace("[", "(").replace("]", ")")

# Parse SMARTS to obtain molecule objects
reactant_mol = Chem.MolFromSmarts(reactant_smarts)
product_mol = Chem.MolFromSmarts(product_smarts)

print("Done")
