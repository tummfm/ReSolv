"""Neural network models for potential energy and molecular property
prediction.
 """
from ml_collections import ConfigDict

def initialize_nequip_cfg(n_species, r_cut) -> ConfigDict:
    """Initialize configuration based on values of original paper."""

    config = ConfigDict()

    # network
    config.graph_net_steps = 5  #2  #(3-5 work best -> Paper)
    config.nonlinearities = {'e': 'raw_swish', 'o': 'tanh'}
    config.use_sc = True
    config.n_elements = n_species
    # config.hidden_irreps = '64x0o + 64x0e + 64x1o + 64x1e'  # '4x0e + 2x1e'    # If parity=True contains o, else only e
    # (irreducible representation of hidden/latent features) (Removing odd often returns same results -> Paper) (32
    # would be a good default. For accuracy increase, for performance decrease.)
    # config.hidden_irreps = '64x0e + 64x1e'
    config.hidden_irreps = '64x0e + 64x1e + 64x2e'
    # config.sh_irreps = '1x0e + 1x1o'  # l=1 and parity=True, # '1x0e + 1x1e'    # Checked implementation in Pytorch (irreducible representations on the edges)
    # config.sh_irreps = '1x0e + 1x1e'
    config.sh_irreps = '1x0e + 1x1e + 1x2e'
    config.num_basis = 8
    config.r_max = r_cut
    config.radial_net_nonlinearity = 'raw_swish'
    config.radial_net_n_hidden = 64  # 8  # Smaller is faster # 64
    config.radial_net_n_layers = 3

    # Setting dependend on dataset

    # QM7x all data
    config.n_neighbors = 15
    config.shift = 0.
    config.scale = 1.
    config.scalar_mlp_std = 4
    return config




def initialize_nequip_cfg_MaxSetup(n_species, r_cut) -> ConfigDict:
    """Initialize configuration based on values of original paper."""

    config = ConfigDict()

    # Information on hyperparameters
    # 1. Cutoff is very important - For MD17 papers employ cutoff = 4 Å
    # Further potential changes:
        # 1. Increase hidden layers
        # 2. Increase number of basis functions

    # network
    config.graph_net_steps = 5  #2  #(3-5 work best -> Paper)
    config.nonlinearities = {'e': 'raw_swish', 'o': 'tanh'}
    config.use_sc = True
    config.n_elements = n_species
    config.hidden_irreps = '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e'  # l_max=2, parity=True
    config.sh_irreps = '1x0e+1x1o+1x2e'     # l_max=2, parity=True
    config.num_basis = 8
    config.r_max = r_cut
    config.radial_net_nonlinearity = 'raw_swish'
    config.radial_net_n_hidden = 64  # 8  # Smaller is faster # 64
    config.radial_net_n_layers = 3

    # Setting dependend on dataset
    # QM7x all data
    config.n_neighbors = 15
    config.shift = 0.
    config.scale = 1.
    config.scalar_mlp_std = 4.
    return config
