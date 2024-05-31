grep -v HETATM 1pgb.pdb > 1pgb_protein_tmp.pdb
grep -v CONECT 1pgb_protein_tmp.pdb > 1pgb_protein.pdb
grep MISSING 1pgb.pdb
gmx pdb2gmx -f 1pgb_protein.pdb -o 1pgb_processed.pdb -water tip3p -ff "charmm27"