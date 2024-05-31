# Molecular Dynamic Simulations with LAMMPS and GROMACS

Code to run LAMMPS simulation via console or python and read the output files using ASE.

LAMMPS provides two different output formats. A dump file (often ```.trr```), which includes per atom snapshots of the simulation.
A log file (often ```log.lammps```), which provides thermodynamic data per timestep. Here we use LAMMPS to run simulations of atomistic water and lennard jones particles.

In order to simulate proteins in solvents, we use GROMACS. The simulations can simple be executed from the console and produce a trajectory file (```.trr```), binary file with simulation details (```.tpr```), energy file (```.edr```), and a structure file (```.gro```).

The installation assumes, that we have created a virtual environment ```venv``` and installed the packages from myjaxmd.

Required python packages:<br>
```
sudo apt-get install python3.8-devk
sudo apt install python3.8-distutils
```

## Installation

1. Activate virtual env
```
source venv/bin/active
```
2. Install extra packages<br/>

add -r file here, update requirements, order, jax pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
install chemtrain 

```
pip install ase
pip install MDAnalysis
pip install seaborn
```
In order to use LAMMPS with python we need to download the source with git:
```
git clone -b stable https://github.com/lammps/lammps.git mylammps
```
Then in mylammps we create a build directory and build the lammps python library and a shared lammps library in our virtual environment (with the lammps packages: ```Kspace```, ```Molecule```, ```Rigid```):
```
cd mylammps               
mkdir build; cd build

cmake -D BUILD_SHARED_LIBS=on -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on -D CMAKE_INSTALL_PREFIX:PATH="/home/usrname/myjaxmd/venv/" -D PYTHON_EXECUTABLE="/home/usrname/myjaxmd/venv/bin/python/" ../cmake

cmake --build .

make install-python

```

Find packages here: https://docs.lammps.org/Packages_list.html. We can now import the ```lammps``` Python module in our files.


## Run LAMMPS

To run our lammps inputfile ```in.LJfluid``` on 8 processors use:
```
mpirun -n 8 lmp_stable -in in.LJfluid
```

This will create a trajectory file ```LJfluid.trj``` and a ```log.lammps``` file.

## Run GROMACS

Before running our simulations, we need to activate GROMACS:
```
source /usr/local/gromacs/bin/GMXRC
```
We start with an initial protein structure (```C5.pdb```) and then select a force field (ex. ```Amber03```) and water solvent model (```tip3p```).
```
gmx pdb2gmx -f C5.pdb -water tip3p -o initial.gro
```
We edit the box size
```
gmx editconf -f initial.gro -bt cubic -box 2.7 2.7 2.7 -o cubic.gro
```
Then we add the water to the box
```
gmx solvate -cp cubic.gro -cs spc216.gro -p topol.top -o water.gro
```
Finally we can perform our calculations (MD or energy minimization). Find more infos in ```/aa_simulations/alanine_dipeptide/setupGromacsProtein.txt```. In general, we create the binary file of our simulation with ```gmx grompp``` and start with ```gmx mdrun```.
```
gmx grompp -f mdfile.mdp -c start.gro -t start.cpt -p topol.top -o md.tpr

gmx mdrun -v -deffnm md -nt 10 -cpt 30
```
This will run GROMACS on 10 processors and checkpoint every 30 minutes. The results are contained in the trajectory file ```md.trr``` and the structure file ```md.gro```.

## Preprocessing datasets with Python

In order to read out the simulation data and create numpy arrays usabels for the NN training, we run the ```create_dataset.py``` file.

## Analysis of the CG simulations

To analyze the water and protein simulations, use ```analysis_water.py``` or ```analysis_protein.py```. This will create plots in ```/plots/postprocessing```.