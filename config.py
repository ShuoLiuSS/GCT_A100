import torch

class Config:
    def __init__(self):
        # Model parameters
        self.d_model = 512
        self.n_layers = 2
        self.heads = 4
        self.dropout = 0.1
        self.batchsize = 2
        self.max_strlen = 80
        self.latent_dim = 32
        self.cond_dim = 3
        self.use_cond2dec = False
        self.use_cond2lat = True
        
        # Training parameters
        self.epochs = 10
        self.lr = 0.0001
        self.lr_beta1 = 0.9
        self.lr_beta2 = 0.98
        
        # Device configuration
        self.device = 'cpu'
        
        # Required attributes for generation
        self.print_model = False
        self.load_weights = None
        self.k = 4
        self.src_pad = 0
        self.trg_pad = 0
