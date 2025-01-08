# ReSolv

Code of the paper "Predicting solvation free energies with an implicit solvent machine learning potential".

The scripts to execute the code are the following: 

1. **Train an U_vac MLP on the QM7x dataset**: examples/FreeEnergyScripts/QM7x scripts/Nequip_QM7x_Prior.py

2. **Precomptue trajectories**: examples/FreeEnergyScripts/QM7x scripts/TrainFreeEnergy/BAR_HFE_trajectory_generator.py \
   which can be called from the bash script BAR_HFE_generate_trajectories.sh within the same directory.

3. **Train the BAR-HFE model**: examples/FreeEnergyScripts/QM7x scripts/TrainFreeEnergy/BAR_HFE_training.py, \
   which can be called from the bash script BAR_HFE_train.sh within the same directory.

## Installation

1. Create virtual environment with Python 3.10 and activate it
```
virtualenv -p /usr/bin/python3.10 venv
```
2. Install all required packages: <br/>
```
pip install swig, gym[box2d]
pip install -e .[all]
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "jax[cuda]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install h5py, rdkit, scipy==1.12, imageio, networkx, dill
```
or use the requirements.txt file. If you use the requirements.txt, you still need to install the local chemtrain and 
jax with cuda support (second last line above):
```
1. pip install -r requirements.txt
2. pip install "jax[cuda]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
3. Install chemtrain from the repository
```


# Additional data and pre-trained models to run the code
## BAR - Learn solvation free energies with ReSolv
1. Add FreeEnergyScripts/QM7x Scripts/FreeSolvDB/database.pickle. Already added in the repository.
2. Add FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/precomputed_trajectories/*.npy for precomputed trajectories. 
   Already added in the repository.
3. Add FreeEnergyScripts/QM7x Scripts/TrainFreeEnergy/checkpoints/080524_t_prod_250ps_t_equil_50ps_iL1e-06_lrd0.1_epochs500_seed7_train_389mem_0.97_epoch499.pkl 
   for the trained U_wat model. Already added in the repository.
4. Add FreeEnergyScripts/savedTrainers/080524_t_prod_250ps_t_equil_50ps_iL1e-06_lrd0.1_epochs500_seed7_train_389mem_0.97_epoch499.pkl 
   for the trained U_vac model. Already added in the repository.

## QM7-x Force Matching
1. Use FreeEnergyScripts/QM7x Scripts/preprocess_qm7x.py to generate your own shuffled data. 

## Contact
For questions, you can reach out to s.roecken@tum.de
## Citation
If you use our paper, the corresponding citation is:
```
@article{10.1063/5.0235189,
    author = {Röcken, Sebastien and Burnet, Anton F. and Zavadlav, Julija},
    title = {Predicting solvation free energies with an implicit solvent machine learning potential},
    journal = {The Journal of Chemical Physics},
    volume = {161},
    number = {23},
    pages = {234101},
    year = {2024},
    month = {12},
    issn = {0021-9606},
    doi = {10.1063/5.0235189},
    url = {https://doi.org/10.1063/5.0235189},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0235189/20300206/234101\_1\_5.0235189.pdf},
}


```
