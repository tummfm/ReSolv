import numpy as np
import mdtraj
from sklearn.decomposition import PCA
from math import pi

################################################
readMD = True

################################################
if (readMD):
    traj = mdtraj.load_trr("MD/100ns/md.trr", top="MD/100ns/md.gro", stride=None, atom_indices=None, frame=None)
    print(traj)
    print(traj.xyz.shape)

    phi = mdtraj.compute_phi(traj, periodic=True, opt=True)
    psi = mdtraj.compute_psi(traj, periodic=True, opt=True)

    outfile = open("MDAnalysis/rama_test.txt", "w")
    for kk in range(len(phi[1])):
        outfile.write("%.15g\t%.15g\n"%(phi[1][kk]*180/pi, psi[1][kk]*180/pi))
    outfile.close()


else:
    traj = mdtraj.load_trr("NMtrr/NM.trr", top="MDtrr/initialReduce.gro", stride=None, atom_indices=None, frame=None)

    phi = mdtraj.compute_phi(traj, periodic=False, opt=True)
    psi = mdtraj.compute_psi(traj, periodic=False, opt=True)

    outfile = open("NMAnalysis/rama_test.txt", "w")
    for kk in range(len(phi[1])):
        outfile.write("%.15g\t%.15g\n"%(phi[1][kk]*180/pi, psi[1][kk]*180/pi))
    outfile.close()