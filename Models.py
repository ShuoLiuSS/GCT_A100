import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy
import numpy as np

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cond_dim = opt.cond_dim
        self.d_model = d_model
        self.embed_sentence = Embedder(vocab_size, d_model)
        self.embed_cond2enc = nn.Linear(opt.cond_dim, d_model*opt.cond_dim)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

        self.fc_mu = nn.Linear(d_model, opt.latent_dim)
        self.fc_log_var = nn.Linear(d_model, opt.latent_dim)

    def forward(self, src, cond_input, mask):
        # Move inputs to the correct device
        device = next(self.parameters()).device
        src = src.to(device)
        cond_input = cond_input.to(device)
        mask = mask.to(device)
        
        cond2enc = self.embed_cond2enc(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
        x = self.embed_sentence(src)
        x = torch.cat([cond2enc, x], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return self.sampling(mu, log_var), mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)


class Decoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cond_dim = opt.cond_dim
        self.d_model = d_model
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        
        # Initialize embeddings and layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.fc_z = nn.Linear(opt.latent_dim, d_model)
        
        # Initialize conditional embeddings
        if self.use_cond2dec:
            self.embed_cond2dec = nn.Linear(opt.cond_dim, d_model * opt.cond_dim)
        if self.use_cond2lat:
            self.embed_cond2lat = nn.Linear(opt.cond_dim, d_model * opt.cond_dim)
            
        # Initialize decoder layers
        self.layers = get_clones(DecoderLayer(opt, d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, cond_input, src_mask, trg_mask):
        # Ensure all inputs are on the correct device
        device = trg.device
        
        # Process inputs
        x = self.embed(trg)
        e_outputs = self.fc_z(e_outputs)
        
        if self.use_cond2dec:
            cond2dec = self.embed_cond2dec(cond_input)
            cond2dec = cond2dec.view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)
            
        if self.use_cond2lat:
            cond2lat = self.embed_cond2lat(cond_input)
            cond2lat = cond2lat.view(cond_input.size(0), cond_input.size(1), -1)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)
            
        x = self.pe(x)
        
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
            
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, opt, src_vocab, trg_vocab):
        super().__init__()
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.encoder = Encoder(opt, src_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
        self.decoder = Decoder(opt, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
        self.out = nn.Linear(opt.d_model, trg_vocab)
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)
    def forward(self, src, trg, cond, src_mask, trg_mask):
        # Move inputs to the correct device
        device = next(self.parameters()).device
        src = src.to(device)
        trg = trg.to(device)
        cond = cond.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)
        if trg_mask is not None:
            trg_mask = trg_mask.to(device)

        z, mu, log_var = self.encoder(src, cond, src_mask)
        d_output = self.decoder(trg, z, cond, src_mask, trg_mask)
        output = self.out(d_output)
        if self.use_cond2dec:
            output_prop, output_mol = self.prop_fc(output[:, :3, :]), output[:, 3:, :]
        else:
            output_prop, output_mol = torch.zeros(output.size(0), 3, 1).to(device), output
        return output_prop, output_mol, mu, log_var, z

def get_model(opt, src_vocab, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(opt, src_vocab, trg_vocab)
    
    if opt.print_model:
        print("model structure:\n", model)

    # Move model to device
    device = opt.device
    model = model.to(device)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        weights = torch.load(f'{opt.load_weights}/model_weights', map_location=device)
        model.load_state_dict(weights)
        print("Model loaded successfully")

    return model
