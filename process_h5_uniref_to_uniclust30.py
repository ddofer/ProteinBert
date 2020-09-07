#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import h5py
import os
# from pwas.shared_utils.util import start_log, log, get_chunk_intervals


## We could add auxiliary taregst as well at this point - e.g. length, AA, 2d structure #, PI.. - https://biopython.org/wiki/ProtParam
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from utils import *


# In[2]:


### list of all files containg "Test set" sequences.
# WE could work with fastas (and look at accession IDs) or our processed CSV sequences (and work on the full sequence); 
# I work on the latter for maximum compatability, even if it is slower. (And TAPE comes in JSON)
TEST_SET_FILES = ["data/TAPE_benchmarks/fluorescence.test.csv", # probably unnecessary
                  "data/TAPE_benchmarks/remote_homology.test.csv",
                  "data/TAPE_benchmarks/secondary_structure.cb513.csv",
                  "data/TAPE_benchmarks/stability.test.csv",
                  "data/scop.dataset.test.csv",
                  "data/signalP_binary.dataset.test.csv"
                 ]

uniclust30_path = "data/uniclust/uniclust30_2018_08_consensus.fasta"

INPUT_H5_FILE_PATH = 'data/dataset-100000.h5'
# filename = INPUT_H5_FILE_PATH


# #### Sequences from test set that we want to filter/remove from our data
# * May need to do blast/cd-hit search for redundnacy. If using uniclust30, then can just drop by exact match

# In[3]:


test_seqs = set([])
for f in TEST_SET_FILES:
    test_seqs.update(pd.read_csv(f,usecols=["text"])[:]["text"])
print(len(test_seqs))


# ##### read uniclust30 data - into memory
# * filter by test set sequences (i.e remove any of them in uniclust 30)
# * Then keep only uniref90 files that are in the (filtered) uniclust sequences/data

# In[4]:


uniclust = fasta_to_df(uniclust30_path,seq_only=True) # we could also get the header and extract accession Id from it

print("sequences in raw uniclust",len(uniclust))
uniclust = uniclust.loc[~uniclust.iloc[:,0].isin(test_seqs)]
uniclust = set(list(uniclust[0]))
print("after filtering out test set sequences, Uniclust size:",len(uniclust))


# #### read h5py file / Uniref90
# * then filter each row in it, by membership in (filtered) uniclust

# In[5]:


f = h5py.File(INPUT_H5_FILE_PATH,'r')
f


# In[6]:


hdf_cols = list(f.keys())
hdf_cols


# In[7]:


f["included_annotation_indices"][0:4]


# In[8]:


len(f['seq_lengths'])


# In[9]:


len(f['included_annotation_indices'])


# * __Warning__ - this processing  currently drops `included_annotation_indices` !!!

# In[10]:


hdf_1d_cols = [
#  'included_annotation_indices',  # there are less annotations than sequences, this causes error - #TODO: FIX!
 'seq_lengths',
 'seqs',
 'uniprot_ids']
df = pd.DataFrame(columns=list(f.keys()))

df['annotation_masks'] = list(f['annotation_masks'][:]) # column of lists

for c in hdf_1d_cols:
    print(c)
    df[c] = f[c][:]
df


# In[11]:


print(df.shape[0])
df = df.loc[df["seqs"].isin(uniclust)]
print("after filtering",df.shape[0])


# In[ ]:




