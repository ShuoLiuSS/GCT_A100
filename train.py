import os
import argparse
import time
import torch
import numpy as np
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import dill as pickle
import pandas as pd
from calProp import calcProperty
import csv
import timeit

def yesno(response):
    """Validates yes/no input."""
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response
	

def KLAnnealer(opt, epoch):
    beta = opt.KLA_ini_beta + opt.KLA_inc_beta * ((epoch + 1) - opt.KLA_beg_epoch)
    return beta

def loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var):
    RCE_mol = F.cross_entropy(preds_mol.contiguous().view(-1, preds_mol.size(-1)), ys_mol, ignore_index=opt.trg_pad, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if opt.use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='sum')
        loss = RCE_mol + RCE_prop + beta * KLD
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_mol + beta * KLD
    return loss, RCE_mol, RCE_prop, KLD

def train_model(model, opt):
   """Train the model using the provided options."""
   print("training model...")
   
   # Setup device and move model
   device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
   model = model.to(device)
   model.train()

   # Initialize timing and checkpointing
   start = time.time()
   if opt.checkpoint > 0:
       cptime = time.time()

   # History tracking lists
   history_epoch, history_beta, history_lr = [], [], []
   history_total_loss, history_RCE_mol_loss, history_RCE_prop_loss, history_KLD_loss = [], [], [], []
   history_total_loss_te, history_RCE_mol_loss_te, history_RCE_prop_loss_te, history_KLD_loss_te = [], [], [], []

   # Initialize training parameters
   beta = 0
   current_step = 0

   # Training loop
   for epoch in range(opt.epochs):
       # Reset epoch metrics 
       total_loss, RCE_mol_loss, RCE_prop_loss, KLD_loss = 0, 0, 0, 0
       total_loss_te, RCE_mol_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0
       total_loss_accum_te, RCE_mol_loss_accum_te, RCE_prop_loss_accum_te, KLD_loss_accum_te = 0, 0, 0, 0
       accum_train_printevery_n, accum_test_n, accum_test_printevery_n = 0, 0, 0

       # Progress indicator
       if not opt.floyd:
           print("     {TR}   %dm: epoch %d [%s]  %d%%  loss = %s" %\
           ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

       # Save checkpoint if needed
       if opt.checkpoint > 0:
           torch.save(model.state_dict(), 'weights/model_weights')

       # KL annealing
       if opt.use_KLA and epoch + 1 >= opt.KLA_beg_epoch and beta < opt.KLA_max_beta:
           beta = KLAnnealer(opt, epoch)
       else:
           beta = 1

       # Training batch loop
       for i, batch in enumerate(opt.train):
           current_step += 1

           # Move batch data to device
           src = batch.src.transpose(0, 1).to(device)
           trg = batch.trg.transpose(0, 1).to(device)
           trg_input = trg[:, :-1]
           cond = torch.stack([batch.logP, batch.tPSA, batch.QED]).transpose(0, 1).to(device)

           # Create masks and move to device
           src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
           if src_mask is not None:
               src_mask = src_mask.to(device)
           if trg_mask is not None:
               trg_mask = trg_mask.to(device)

           # Forward pass
           preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
           ys_mol = trg[:, 1:].contiguous().view(-1)
           ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

           # Compute loss and backpropagate
           opt.optimizer.zero_grad()
           loss, RCE_mol, RCE_prop, KLD = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)
           loss.backward()
           opt.optimizer.step()

           # Learning rate scheduling
           if opt.lr_scheduler == "SGDR":
               opt.sched.step()
           elif opt.lr_scheduler == "WarmUpDefault":
               head = np.float(np.power(np.float(current_step), -0.5))
               tail = np.float(current_step) * np.power(np.float(opt.lr_WarmUpSteps), -1.5)
               lr = np.float(np.power(np.float(opt.d_model), -0.5)) * min(head, tail)
               for param_group in opt.optimizer.param_groups:
                   param_group['lr'] = lr

           # Get current learning rate
           for param_group in opt.optimizer.param_groups:
               current_lr = param_group['lr']

           # Update metrics
           total_loss += loss.item()
           RCE_mol_loss += RCE_mol.item()
           RCE_prop_loss += RCE_prop.item()
           KLD_loss += KLD.item()
           accum_train_printevery_n += len(batch)

           # Print progress and save history
           if (i + 1) % opt.printevery == 0:
               p = int(100 * (i + 1) / opt.train_len)
               avg_loss = total_loss / accum_train_printevery_n
               avg_RCE_mol_loss = RCE_mol_loss / accum_train_printevery_n
               avg_RCE_prop_loss = RCE_prop_loss / accum_train_printevery_n
               avg_KLD_loss = KLD_loss / accum_train_printevery_n

               # Save history at intervals
               if (i + 1) % opt.historyevery == 0:
                   history_epoch.append(epoch + 1)
                   history_beta.append(beta)
                   history_lr.append(current_lr)
                   history_total_loss.append(avg_loss)
                   history_RCE_mol_loss.append(avg_RCE_mol_loss)
                   history_RCE_prop_loss.append(avg_RCE_prop_loss)
                   history_KLD_loss.append(avg_KLD_loss)

               # Print progress
               progress_str = "     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f"
               progress_args = ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), 
                              "".join(' '*(20-(p//5))), p, avg_loss, avg_RCE_mol_loss, 
                              avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr)
               
               if not opt.floyd:
                   print(progress_str % progress_args, end='\r')
               else:
                   print(progress_str % progress_args)

               # Reset accumulators
               accum_train_printevery_n = 0
               total_loss = RCE_mol_loss = RCE_prop_loss = KLD_loss = 0

           # Save checkpoint if needed
           if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
               torch.save(model.state_dict(), 'weights/model_weights')
               cptime = time.time()

       # End of epoch reporting
       print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" %\
       ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 
        100, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr))

       # Testing phase
       if opt.imp_test:
           model.eval()
           if not opt.floyd:
               print("     {TE}   %dm:         [%s]  %d%%  loss = %s" %\
               ((time.time() - start)//60, "".join(' '*20), 0, '...'), end='\r')

           # Test loop with no gradient computation
           with torch.no_grad():
               for i, batch in enumerate(opt.test):
                   # Move batch data to device
                   src = batch.src.transpose(0, 1).to(device)
                   trg = batch.trg.transpose(0, 1).to(device)
                   trg_input = trg[:, :-1]
                   cond = torch.stack([batch.logP, batch.tPSA, batch.QED]).transpose(0, 1).to(device)

                   # Forward pass
                   src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
                   if src_mask is not None:
                       src_mask = src_mask.to(device)
                   if trg_mask is not None:
                       trg_mask = trg_mask.to(device)
                       
                   preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
                   ys_mol = trg[:, 1:].contiguous().view(-1)
                   ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

                   # Compute test loss
                   loss_te, RCE_mol_te, RCE_prop_te, KLD_te = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)

                   # Update test metrics
                   total_loss_te += loss_te.item()
                   RCE_mol_loss_te += RCE_mol_te.item()
                   RCE_prop_loss_te += RCE_prop_te.item()
                   KLD_loss_te += KLD_te.item()
                   total_loss_accum_te += loss_te.item()
                   RCE_mol_loss_accum_te += RCE_mol_te.item()
                   RCE_prop_loss_accum_te += RCE_prop_te.item()
                   KLD_loss_accum_te += KLD_te.item()

                   accum_test_n += len(batch)
                   accum_test_printevery_n += len(batch)

                   # Print test progress
                   if (i + 1) % opt.printevery == 0:
                       p = int(100 * (i + 1) / opt.test_len)
                       avg_loss_te = total_loss_te / accum_test_printevery_n
                       avg_RCE_mol_loss_te = RCE_mol_loss_te / accum_test_printevery_n
                       avg_RCE_prop_loss_te = RCE_prop_loss_te / accum_test_printevery_n
                       avg_KLD_loss_te = KLD_loss_te / accum_test_printevery_n

                       test_progress_str = "     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f"
                       test_progress_args = ((time.time() - start)//60, "".join('#'*(p//5)), 
                                          "".join(' '*(20-(p//5))), p, avg_loss_te, avg_RCE_mol_loss_te, 
                                          avg_RCE_prop_loss_te, avg_KLD_loss_te, beta)

                       if not opt.floyd:
                           print(test_progress_str % test_progress_args, end='\r')
                       else:
                           print(test_progress_str % test_progress_args)

                       # Reset test accumulators
                       accum_test_printevery_n = 0
                       total_loss_te = RCE_mol_loss_te = RCE_prop_loss_te = KLD_loss_te = 0

               # Final test reporting
               print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f\n" % \
                     ((time.time() - start)//60, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 
                      100, avg_loss_te, avg_RCE_mol_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, beta))

           # Update test history at the end of epoch
           if epoch == 0:
               for j in range(len(history_epoch)):
                   history_total_loss_te.append("")
                   history_RCE_mol_loss_te.append("")
                   history_RCE_prop_loss_te.append("")
                   history_KLD_loss_te.append("")

           history_total_loss_te.append(total_loss_accum_te/len(opt.test.dataset))
           history_RCE_mol_loss_te.append(RCE_mol_loss_accum_te/len(opt.test.dataset))
           history_RCE_prop_loss_te.append(RCE_prop_loss_accum_te/len(opt.test.dataset))
           history_KLD_loss_te.append(KLD_loss_accum_te/len(opt.test.dataset))

       # Update history and save after test metrics are updated
       history_epoch.append(epoch+1)
       history_beta.append(beta)
       history_lr.append(current_lr)
       history_total_loss.append(avg_loss)
       history_RCE_mol_loss.append(avg_RCE_mol_loss)
       history_RCE_prop_loss.append(avg_RCE_prop_loss)
       history_KLD_loss.append(avg_KLD_loss)

       if opt.imp_test:
            # First, ensure all history arrays are the same length
            max_len = max(len(history_epoch), len(history_total_loss), len(history_total_loss_te))
            
            # Pad test arrays to match max_len
            while len(history_total_loss_te) < max_len:
                history_total_loss_te.append("")
                history_RCE_mol_loss_te.append("")
                history_RCE_prop_loss_te.append("")
                history_KLD_loss_te.append("")
            
            # Now append the new test metrics
            history_total_loss_te[-1] = total_loss_accum_te/len(opt.test.dataset)
            history_RCE_mol_loss_te[-1] = RCE_mol_loss_accum_te/len(opt.test.dataset)
            history_RCE_prop_loss_te[-1] = RCE_prop_loss_accum_te/len(opt.test.dataset)
            history_KLD_loss_te[-1] = KLD_loss_accum_te/len(opt.test.dataset)

        # Update history and save after test metrics are updated
       history_epoch.append(epoch+1)
       history_beta.append(beta)
       history_lr.append(current_lr)
       history_total_loss.append(avg_loss)
       history_RCE_mol_loss.append(avg_RCE_mol_loss)
       history_RCE_prop_loss.append(avg_RCE_prop_loss)
       history_KLD_loss.append(avg_KLD_loss)

       # Create history dictionary
       history_dict = {
           "epochs": history_epoch,
           "beta": history_beta,
           "lr": history_lr,
           "total_loss": history_total_loss,
           "RCE_mol_loss": history_RCE_mol_loss,
           "RCE_prop_loss": history_RCE_prop_loss,
           "KLD_loss": history_KLD_loss
       }
       
       if opt.imp_test:
           # Ensure test arrays match length of training arrays
           while len(history_total_loss_te) < len(history_total_loss):
               history_total_loss_te.append("")
               history_RCE_mol_loss_te.append("")
               history_RCE_prop_loss_te.append("")
               history_KLD_loss_te.append("")
           
           history_dict.update({
               "total_loss_te": history_total_loss_te,
               "RCE_mol_loss_te": history_RCE_mol_loss_te,
               "RCE_prop_loss_te": history_RCE_prop_loss_te,
               "KLD_loss_te": history_KLD_loss_te
           })

       # Save history
       history = pd.DataFrame(history_dict)
       history.to_csv(f'trHist_lat={opt.latent_dim}_epo={opt.epochs}_{time.strftime("%Y%m%d")}.csv', index=True)


def promptNextAction(model, opt, SRC, TRG, robustScaler):
    """Save results to w_trained folder and exit."""
    # Create or update w_trained folder
    dst = 'w_trained'
    os.makedirs(dst, exist_ok=True)
    
    # Save model weights and other files
    print(f"Saving weights and files to {dst}/...")
    torch.save(model.state_dict(), f'{dst}/model_weights')
    pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
    joblib.dump(robustScaler, f'{dst}/scaler.pkl')
    print(f"Files saved to {dst}/")
    print("Training ends")
    

def main():
   # 1. Argument Parsing
   parser = argparse.ArgumentParser()
   # Data settings
   parser.add_argument('-imp_test', type=bool, default=True)
   parser.add_argument('-src_data', type=str, default='data/moses/train.txt')
   parser.add_argument('-src_data_te', type=str, default='data/moses/test.txt')
   parser.add_argument('-trg_data', type=str, default='data/moses/train.txt')
   parser.add_argument('-trg_data_te', type=str, default='data/moses/test.txt')
   parser.add_argument('-lang_format', type=str, default='SMILES')
   parser.add_argument('-calProp', type=bool, default=True)

   # Learning hyperparameters
   parser.add_argument('-epochs', type=int, default=10)
   parser.add_argument('-no_cuda', action='store_true')
   parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
   parser.add_argument('-lr_WarmUpSteps', type=int, default=8000)
   parser.add_argument('-lr', type=float, default=0.0001)
   parser.add_argument('-lr_beta1', type=float, default=0.9)
   parser.add_argument('-lr_beta2', type=float, default=0.98)
   parser.add_argument('-lr_eps', type=float, default=1e-9)

   # KL Annealing
   parser.add_argument('-use_KLA', type=bool, default=True)
   parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
   parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
   parser.add_argument('-KLA_max_beta', type=float, default=1.0)
   parser.add_argument('-KLA_beg_epoch', type=int, default=1)

   # Network structure
   parser.add_argument('-use_cond2dec', type=bool, default=False)
   parser.add_argument('-use_cond2lat', type=bool, default=True)
   parser.add_argument('-latent_dim', type=int, default=128)
   parser.add_argument('-cond_dim', type=int, default=3)
   parser.add_argument('-d_model', type=int, default=512)
   parser.add_argument('-n_layers', type=int, default=6)
   parser.add_argument('-heads', type=int, default=8)
   parser.add_argument('-dropout', type=float, default=0.3)
   parser.add_argument('-batchsize', type=int, default=256)
   parser.add_argument('-max_strlen', type=int, default=80)

   # History and saving
   parser.add_argument('-verbose', type=bool, default=False)
   parser.add_argument('-save_folder_name', type=str, default='saved_model')
   parser.add_argument('-print_model', type=bool, default=False)
   parser.add_argument('-printevery', type=int, default=5)
   parser.add_argument('-historyevery', type=int, default=5)
   parser.add_argument('-load_weights', type=str, default=None)
   parser.add_argument('-create_valset', action='store_true')
   parser.add_argument('-floyd', action='store_true')
   parser.add_argument('-checkpoint', type=int, default=0)

   opt = parser.parse_args()

   # 2. Device Setup and Validation Checks
   opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
   print(f"Using device: {opt.device}")

   if opt.historyevery % opt.printevery != 0:
       raise ValueError(f"historyevery must be a multiple of printevery: {opt.historyevery} % {opt.printevery} != 0")

   # 3. Data Loading
   read_data(opt)

   # 4. Property Calculation/Loading
   if opt.calProp:
       PROP, PROP_te = calcProperty(opt)
   else:
       PROP = pd.read_csv("data/moses/prop_temp.csv")
       PROP_te = pd.read_csv("data/moses/prop_temp_te.csv")

   # 5. Create Fields
   SRC, TRG = create_fields(opt)

   # 6. Property Processing
   # Get property bounds
   opt.max_logP, opt.min_logP = PROP["logP"].max(), PROP["logP"].min()
   opt.max_tPSA, opt.min_tPSA = PROP["tPSA"].max(), PROP["tPSA"].min()
   opt.max_QED, opt.min_QED = PROP_te["QED"].max(), PROP_te["QED"].min()

   # Scale properties
   robustScaler = RobustScaler()
   PROP_scaled = robustScaler.fit_transform(PROP)
   PROP_te_scaled = robustScaler.transform(PROP_te)
   
   # Create scaled DataFrames with column names
   PROP = pd.DataFrame(PROP_scaled, columns=['logP', 'tPSA', 'QED'])
   PROP_te = pd.DataFrame(PROP_te_scaled, columns=['logP', 'tPSA', 'QED'])

   # 7. Create Datasets
   opt.train = create_dataset(opt, SRC, TRG, PROP, tr_te='tr')
   opt.test = create_dataset(opt, SRC, TRG, PROP_te, tr_te='te')

   # 8. Model Setup
   model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
   total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f"\n# of trainable parameters: {total_trainable_params}")

   # 9. Optimizer Setup
   opt.optimizer = torch.optim.Adam(model.parameters(), 
                                  lr=opt.lr, 
                                  betas=(opt.lr_beta1, opt.lr_beta2), 
                                  eps=opt.lr_eps)
   
   if opt.lr_scheduler == "SGDR":
       opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

   # 10. Weight Management
   if opt.checkpoint > 0:
       print(f"Model weights will be saved every {opt.checkpoint} minutes and at end of epoch to directory weights/")
   
   if opt.load_weights is not None and opt.floyd is not None:
       os.makedirs('weights', exist_ok=True)
       pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
       pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

   # 11. Training
   train_model(model, opt)

   # 12. Post-training Actions
   if not opt.floyd:
       promptNextAction(model, opt, SRC, TRG, robustScaler)

if __name__ == "__main__":
   main()

