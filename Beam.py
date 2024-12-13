import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

def init_vars(cond, model, SRC, TRG, toklen, opt, z):
    """Initialize variables for beam search.
    
    Args:
        cond: Condition tensor for molecule generation
        model: The transformer model
        SRC: Source field
        TRG: Target field
        toklen: Token length
        opt: Options/arguments
        z: Latent vector
    """
    device = next(model.parameters()).device
    init_tok = TRG.vocab.stoi['<sos>']

    # Create and move masks to device
    src_mask = (torch.ones(1, 1, toklen) != 0).to(device)
    trg_mask = nopeak_mask(1, opt).to(device)

    # Create and move input tensor to device
    trg_in = torch.LongTensor([[init_tok]]).to(device)
    z = z.to(device)
    cond = cond.to(device)

    # Forward pass through decoder
    if opt.use_cond2dec:
        output_mol = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))[:, 3:, :]
    else:
        output_mol = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))
    
    # Calculate probabilities
    out_mol = F.softmax(output_mol, dim=-1)
    probs, ix = out_mol[:, -1].data.topk(opt.k)
    log_scores = torch.tensor([math.log(prob) for prob in probs.data[0]], 
                            device=device).unsqueeze(0)

    # Initialize outputs
    outputs = torch.zeros(opt.k, opt.max_strlen, device=device, dtype=torch.long)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    # Initialize encoder outputs
    e_outputs = torch.zeros(opt.k, z.size(-2), z.size(-1), device=device)
    e_outputs[:, :] = z[0]

    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    """Find the k best outputs at step i."""
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.tensor([math.log(p) for p in probs.data.view(-1)], 
                           device=outputs.device).view(k, -1)
    log_probs += log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(cond, model, SRC, TRG, toklen, opt, z):
    """Perform beam search for molecule generation.
    
    Args:
        cond: Condition tensor
        model: The transformer model
        SRC: Source field
        TRG: Target field
        toklen: Token length
        opt: Options/arguments
        z: Latent vector
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move inputs to device
    cond = cond.to(device)
    z = z.to(device)

    # Initialize variables
    outputs, e_outputs, log_scores = init_vars(cond, model, SRC, TRG, toklen, opt, z)
    
    # Repeat condition for beam size
    cond = cond.repeat(opt.k, 1)
    
    # Create and move source mask to device
    src_mask = (torch.ones(1, 1, toklen) != 0)
    src_mask = src_mask.repeat(opt.k, 1, 1).to(device)
    
    # Get end token
    eos_tok = TRG.vocab.stoi['<eos>']
    
    # Beam search
    ind = None
    for i in range(2, opt.max_strlen):
        # Create target mask
        trg_mask = nopeak_mask(i, opt)
        trg_mask = trg_mask.repeat(opt.k, 1, 1).to(device)

        # Decode
        if opt.use_cond2dec:
            output_mol = model.out(model.decoder(outputs[:,:i], e_outputs, cond, 
                                               src_mask, trg_mask))[:, 3:, :]
        else:
            output_mol = model.out(model.decoder(outputs[:,:i], e_outputs, cond, 
                                               src_mask, trg_mask))
            
        out_mol = F.softmax(output_mol, dim=-1)
        outputs, log_scores = k_best_outputs(outputs, out_mol, log_scores, i, opt.k)
        
        # Find completed sequences
        ones = (outputs == eos_tok).nonzero()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:
                sentence_lengths[i] = vec[1]

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        # If all sequences are completed
        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    # Return generated sequence
    if ind is None:
        try:
            length = (outputs[0] == eos_tok).nonzero()[0]
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
        except:
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:]])
    
    length = (outputs[ind] == eos_tok).nonzero()[0]
    return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
