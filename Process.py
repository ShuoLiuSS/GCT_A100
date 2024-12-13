import pandas as pd
import torch
import torchtext
from torchtext import data
from Tokenize import moltokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import numpy as np


def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

    if opt.src_data_te is not None:
        try:
            opt.src_data_te = open(opt.src_data_te, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data_te + "' file not found")
            quit()

    if opt.trg_data_te is not None:
        try:
            opt.trg_data_te = open(opt.trg_data_te, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data_te + "' file not found")
            quit()


def create_fields(opt):
    lang_formats = ['SMILES', 'SELFIES']
    if opt.lang_format not in lang_formats:
        print('invalid src language: ' + opt.lang_forma + 'supported languages : ' + lang_formats)

    print("loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer)
    TRG = data.Field(tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))

        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(opt, SRC, TRG, PROP, tr_te):
   """Create dataset and iterator for training or testing.
   
   Args:
       opt: Options/arguments
       SRC: Source field for tokenization 
       TRG: Target field for tokenization
       PROP: DataFrame containing molecular properties (logP, tPSA, QED)
       tr_te: Either 'tr' for training or 'te' for testing
   """
   
   # Create raw data dictionary from source
   if tr_te == "tr":
       print("\n* creating [train] dataset and iterator... ")
       raw_data = {'src': [line for line in opt.src_data], 
                  'trg': [line for line in opt.trg_data]}
   else:  # te
       print("\n* creating [test] dataset and iterator... ")
       raw_data = {'src': [line for line in opt.src_data_te], 
                  'trg': [line for line in opt.trg_data_te]}

   # Create DataFrame and add properties
   df = pd.DataFrame(raw_data, columns=["src", "trg"]) 
   df = pd.concat([df, PROP], axis=1)

   # Mask sequences longer than max_strlen
   if opt.lang_format == 'SMILES':
       mask = ((df['src'].str.len() + opt.cond_dim < opt.max_strlen) & 
              (df['trg'].str.len() + opt.cond_dim < opt.max_strlen))
   df = df.loc[mask]

   # Validate and clean property values
   for col in ['logP', 'tPSA', 'QED']:
       df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

   # Save temporary CSV file and print info
   if tr_te == "tr":
       print("     - # of training samples:", len(df.index))
       print("     - Property value ranges:")
       for col in ['logP', 'tPSA', 'QED']:
           print(f"       {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
       temp_file = "DB_transformer_temp.csv"
   else:
       print("     - # of test samples:", len(df.index))
       print("     - Property value ranges:")
       for col in ['logP', 'tPSA', 'QED']:
           print(f"       {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
       temp_file = "DB_transformer_temp_te.csv"
   
   df.to_csv(temp_file, index=False)

   # Define fields for properties with error handling
   prop_fields = {
       'logP': data.Field(use_vocab=False, sequential=False, dtype=torch.float,
                         preprocessing=lambda x: float(x) if x and str(x).strip() != '' else 0.0),
       'tPSA': data.Field(use_vocab=False, sequential=False, dtype=torch.float,
                         preprocessing=lambda x: float(x) if x and str(x).strip() != '' else 0.0),
       'QED':  data.Field(use_vocab=False, sequential=False, dtype=torch.float,
                         preprocessing=lambda x: float(x) if x and str(x).strip() != '' else 0.0)
   }

   # Combine all fields
   data_fields = [('src', SRC), ('trg', TRG)]
   data_fields.extend([(name, field) for name, field in prop_fields.items()])

   # Create dataset
   try:
       train = data.TabularDataset(
           path=temp_file,
           format='csv',
           fields=data_fields,
           skip_header=True
       )
   except Exception as e:
       print(f"Error creating dataset: {e}")
       print("Sample of problematic data:")
       print(df.head())
       raise e

   if tr_te == "tr":
       # Save token lengths and build vocabulary
       toklenList = [len(vars(example)['src']) for example in train]
       df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
       df_toklenList.to_csv("toklen_list.csv", index=False)

       if opt.verbose:
           print("     - tokenized training sample 0:", vars(train[0]))

       if opt.load_weights is None:
           print("     - building vocab from train data...")
           SRC.build_vocab(train)
           TRG.build_vocab(train)
           
           if opt.verbose:
               print('     - vocab size of SRC:', len(SRC.vocab))
               print('     - vocab size of TRG:', len(TRG.vocab))

           if opt.checkpoint > 0:
               try:
                   os.mkdir("weights")
               except FileExistsError:
                   print("weights folder already exists, run program with -load_weights weights to load them")
                   quit()
               pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
               pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

       opt.src_pad = SRC.vocab.stoi['<pad>']
       opt.trg_pad = TRG.vocab.stoi['<pad>']

   elif tr_te == "te" and opt.verbose:
       print("     - tokenized testing sample 0:", vars(train[0]))

   # Create iterator
   train_iter = MyIterator(
       train, 
       batch_size=opt.batchsize,
       device=opt.device,
       repeat=False,
       sort_key=lambda x: (len(x.src), len(x.trg)),
       batch_size_fn=batch_size_fn,
       train=(tr_te == "tr"),
       shuffle=(tr_te == "tr")
   )

   # Calculate length
   len_iter = MyIterator(
       train,
       batch_size=opt.batchsize,
       device=opt.device,
       repeat=False,
       sort_key=lambda x: (len(x.src), len(x.trg)),
       batch_size_fn=batch_size_fn,
       train=False,
       shuffle=False
   )
   train_len = sum(1 for _ in len_iter)

   if tr_te == "tr":
       opt.train_len = train_len
   else:
       opt.test_len = train_len

   # Clean up temporary files
   try:
       os.remove(temp_file)
   except:
       pass

   return train_iter


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

