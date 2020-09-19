#!/usr/bin/env python
# coding: utf-8

# # Setup
# 
# 
# 
# * note: for sent2vec - an improvement - "A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS"
#      * https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py

# In[87]:


import re
import gzip

from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import sentencepiece as spm

import fasttext

# https://radimrehurek.com/gensim/models/doc2vec.html
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# model = KeyedVectors.load_word2vec_format('../input/en.wiki.bpe.op1000.d25.w2v.bin', binary=True)


# In[88]:


EXTRACT_FASTA_RECORDS = False
TRAIN_SENTENCE_PIECE = False
EXTRACT_H5 = False

EXTRACT_SENTPIECE_TOKENIZED_SEQ = False

TRAIN_FAST_TEXT_UNSUPERVISED = False


# In[89]:


EXTRACT_SENTPIECE_LABELED_TOKENIZED_SEQ = True
LABELED_DATA_FILE_PATH = "labelled_toy_seqs_v1.csv.gz"

TRAIN_LABELED_DATA_FILE_PATH = "labelled_toy_seqs_v1_TRAIN.csv"
TEST_LABELED_DATA_FILE_PATH = "labelled_toy_seqs_v1_TEST.csv"


# In[90]:


# # os.chdir(os.path.normpath(r"C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert"))
# os.chdir(r"C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert")
print(os.curdir)


# # Create a plain text with protein seqs

# > wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz

# In[4]:


# get_ipython().system('cd "C:\\Users\\Dan Ofer\\Desktop\\Stuff\\cafa\\ProteinBert\\data"')


# In[5]:


# os.chdir(r"C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data")
# print(os.curdir)


# In[6]:


# INPUT_FASTA_FILE_PATH = '/cs/phd/nadavb/cafa_project/data/uniref90.fasta.gz'
# CORPUS_TXT_FILE_PATH = '/cs/phd/nadavb/cafa_project/data/seqs_for_sentencepeice_training.txt'

# INPUT_FASTA_FILE_PATH =os.path.normpath('./data/uniclust30_2016_150K_sampledSeq.fasta.gz')
INPUT_FASTA_FILE_PATH =os.path.normpath(r'C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data\uniref_90_go-manual+OR+manual-annotation_transmem+OR+annotation_signal.fasta.gz')

# INPUT_FASTA_FILE_PATH =os.path.normpath(r'C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data\uniclust30_2016_150K_sampledSeq.fasta.gz')
CORPUS_TXT_FILE_PATH = os.path.normpath(r'seqs_for_sentencepeice_training.txt')


#### TODO: Add more output files names here - e.g. for training data


# In[7]:


# TODO: Instead of taking just the first sequences, it could be better to take a random subsample. 
if EXTRACT_FASTA_RECORDS:
    
    N_SEQS = 3123456

    with gzip.open(INPUT_FASTA_FILE_PATH, 'rt') as input_fasta_file, open(CORPUS_TXT_FILE_PATH, 'w') as output_txt_file:
        for i, record in enumerate(SeqIO.parse(input_fasta_file, 'fasta')):

            if N_SEQS is not None and i >= N_SEQS:
                break

            if i % 5000 == 0:
                print(i, end = '\r')

            output_txt_file.write(str(record.seq) + '\n')

    print('Done.')


# # Train a sentencepiece model
# * Note: we will want a larger vocab size here than with our language model. W2V will embed it anyway.

# In[8]:


# # # %cd /cs/phd/nadavb/cafa_project/data
# !cd "C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data"


# In[9]:


# !cd C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data


# In[10]:


VOCAB_SIZE = 150000
# VOCAB_SIZE = 120
N_RESERVED_SYMBOLS = 2 # We want to reserve two symbols: 1) for PADDING, 2) for MASKING.


# In[11]:


if TRAIN_SENTENCE_PIECE:

    # spm.SentencePieceTrainer.Train('--input=%s --model_prefix=protopiece --vocab_size=%d' % (CORPUS_TXT_FILE_PATH, \
    #         VOCAB_SIZE - N_RESERVED_SYMBOLS))


    # spm.SentencePieceTrainer.Train('--input=uniclust30_2016_150K_sampledSeq.txt --model_prefix=protopiece --vocab_size=%d' % (VOCAB_SIZE - N_RESERVED_SYMBOLS))

    spm.SentencePieceTrainer.Train('--input=seqs_for_sentencepeice_training.txt --model_prefix=protopiece --vocab_size=%d' % (VOCAB_SIZE - N_RESERVED_SYMBOLS))


# In[12]:


sp = spm.SentencePieceProcessor()
sp.load('protopiece.model')


# In[13]:


example_seq = 'MRYTVLIALQGALLLLLLIDDGQGQSPYPYPGMPCNSSRQCGLGTCVHSRCAHCSSDGTLCSPEDPTMVWPCCPESSCQLVVG' +               'LPSLVNHYNCLPNQCTDSSQCPGGFGCMTRRSKCELCKADGEACNSPYLDWRKDKECCSGYCHTEARGLEGVCIDPKKIFCTP' +               'KNPWQLAPYPPSYHQPTTLRPPTSLYDSWLMSGFLVKSTTAPSTQEEEDDY'

print(sp.encode_as_pieces(example_seq))
print(sp.encode_as_ids(example_seq))


# # Preprocess our dataset sequences using the trained sentencepiece model

# In[14]:


# DATASET_H5_FILE_PATH = '/cs/phd/nadavb/cafa_project/data/protein_tokens.h5'

DATASET_H5_FILE_PATH = os.path.normpath(r'C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data\protein_tokens.h5')


# In[15]:


# INPUT_FASTA_FILE_PATH = os.path.normpath(r'C:\Users\Dan Ofer\Desktop\Stuff\cafa\ProteinBert\data\uniclust30_2016_150K_sampledSeq.txt')

# INPUT_FASTA_FILE_PATH = os.path.normpath(r".\data\uniclust30_2016_150K_sampledSeq.txt")


# In[16]:


N_SEQS = 1000000

if EXTRACT_H5:
    REP_ID_PATTERN = re.compile(r'RepID=(\S+)')

    with gzip.open(INPUT_FASTA_FILE_PATH, 'rt') as input_fasta_file, h5py.File(DATASET_H5_FILE_PATH, 'w') as h5f:

        h5f_group = h5f.create_group('protein_tokens')
        h5f_rep_id = h5f_group.create_dataset('rep_id', shape = (N_SEQS,), dtype = h5py.string_dtype())
        h5f_tokens = h5f_group.create_dataset('tokens', shape = (N_SEQS,), dtype = h5py.vlen_dtype(np.int16))
        h5f_seq_length = h5f_group.create_dataset('seq_length', shape = (N_SEQS,), dtype = np.int32)
        h5f_n_tokens = h5f_group.create_dataset('n_tokens', shape = (N_SEQS,), dtype = np.int32)

        for i, record in enumerate(SeqIO.parse(input_fasta_file, 'fasta')):

            if N_SEQS is not None and i >= N_SEQS:
                break

            if i % 1000 == 0:
                print(i, end = '\r')

            rep_id, = REP_ID_PATTERN.findall(record.description)
            tokens = sp.encode_as_ids(str(record.seq))
            seq_length = len(record.seq)
            n_tokens = len(tokens)

            h5f_rep_id[i] = rep_id
            h5f_tokens[i] = tokens
            h5f_seq_length[i] = seq_length
            h5f_n_tokens[i] = n_tokens

    print('Done.')


# In[17]:


if EXTRACT_H5:
    with h5py.File(DATASET_H5_FILE_PATH, 'r') as h5f:
        h5f_group = h5f['protein_tokens']
        all_n_tokens = h5f_group['n_tokens'][:]

    fig, ax = plt.subplots(figsize = (10, 4))
    ax.hist(all_n_tokens, bins = 200)
    ax.set_xlabel('# Tokens')
    ax.set_ylabel('# Seqs')


# ## fasttext
# 
# * First we tokenize using sentencepiece
#     * optionally save output to file, if not already saved
# * tran fasttext/word2vec model on text. (unsupervised).
# * +- train supervised model vs compare to k-nn retrieval model 

# * Sentence piece model tokenization of a given file
# * Expect a file with 1 protein sequence per row, no headers or fasta descriptors
# * saves tokenized output (in text format) to new file

# In[18]:


if EXTRACT_SENTPIECE_TOKENIZED_SEQ: 
    text = pd.read_csv(CORPUS_TXT_FILE_PATH,header=None,names=["seq"],
    #                    nrows=140
                      )
    print(text.shape)
    ## orig - .apply(", ".join)
    text["seq"] = text["seq"].apply(sp.encode_as_pieces).apply(" ".join) # apply to get rid of brackets around tokenized text
    display(text.head())

    text["seq"].to_csv("tokenized_sp_fasta.csv",index=False,encoding="utf-8")


# * Train fasttext or word2vec model on tokenized data

# In[19]:


import fasttext
# fasttext.train_supervised
# fasttext.train_unsupervised


# fasttext params
# 
#     input             # training file path (required)
#     model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
#     lr                # learning rate [0.05]
#     dim               # size of word vectors [100]
#     ws                # size of the context window [5]
#     epoch             # number of epochs [5]
#     minCount          # minimal number of word occurences [5]
#     minn              # min length of char ngram [3]
#     maxn              # max length of char ngram [6]
#     neg               # number of negatives sampled [5]
#     wordNgrams        # max length of word ngram [1]
#     loss              # loss function {ns, hs, softmax, ova} [ns]
#     bucket            # number of buckets [2000000]
#     thread            # number of threads [number of cpus]
#     lrUpdateRate      # change the rate of updates for the learning rate [100]
#     t                 # sampling threshold [0.0001]
#     verbose           # verbose [2]

# In[22]:


## todo: hyperparameter search on these variables
EMBEDDING_DIM = 150
WINDOW_SIZE = 5
FT_MODEL = "skipgram" #'cbow'


# In[23]:


if TRAIN_FAST_TEXT_UNSUPERVISED:
    model = fasttext.train_unsupervised("tokenized_sp_fasta.csv", 
    #                                     model=FT_MODEL,
#                                         lr=0.05,
                                        dim=EMBEDDING_DIM,
                                        ws=WINDOW_SIZE,
#                                         epoch=2,
                                        thread=3,  # -1 for all
                                        minn=1,maxn=2,
                                        minCount=4,
                                        verbose=1
#                                        neg=6
                                       )
    print("model trained, saving")
    print(f"model_file_{EMBEDDING_DIM}D_{WINDOW_SIZE}ws_{FT_MODEL}.bin")
#     model.save_model("model_file.bin")

    model.save_model(f"model_file_{EMBEDDING_DIM}D_{WINDOW_SIZE}ws_{FT_MODEL}.bin") ## useful if testing hyperparams


# # Train supervised model
# ## OPT: purely retrieval based/knn model
# 
# * Using toy data, precleaned - first column is sequencer, next columns are labels
#     * note - supervised fasttext treats multilabel as multiclass! 
#     
# * Toy data downloaded from drive : labelled_toy_seqs_v1.csv.gz   , https://drive.google.com/open?id=1mHdkZFv_gNvgWdpzMqKMLCgMOPb84fDU
# 
# 
# * as another strong classification/embedding baseline - can use the "sentence2vec" file code - from https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py 
#     * A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
#     
#     
# * may want to try `scikit-multilearn` package - http://scikit.ml/stratification.html
#     * I use this to get stratified train/test split, across the labels

# In[24]:


from gensim.models.fasttext import FastText


# In[70]:


# ### stratified sampling of dataset into set train/test splits for comparison
# # !pip install scikit-multilearn

# print(LABELED_DATA_FILE_PATH)
# df = pd.read_csv(LABELED_DATA_FILE_PATH)#.sample(n=2134)#.iloc[:,0:5]
# column_names = df.columns
# print(df.shape)

# ## stratified multilabel sampling - http://scikit.ml/stratification.html
# from skmultilearn.model_selection import iterative_train_test_split
# X_train,y_train, X_test, y_test = iterative_train_test_split(X=df.values, y=df.drop(["Sequence"],axis=1).values, test_size = 0.25)

# df_train = pd.DataFrame(X_train,columns=column_names)
# df_test = pd.DataFrame(X_test,columns=column_names)

# display(df_train.head())

# df_train.to_csv("labelled_toy_seqs_v1_TRAIN.csv",index=False)
# df_test.to_csv("labelled_toy_seqs_v1_TEST.csv",index=False)


# In[126]:


# df_train.to_csv("labelled_toy_seqs_v1_TRAIN.csv",index=False)
# df_test.to_csv("labelled_toy_seqs_v1_TEST.csv",index=False)


# ### Apply Sentence piece tokenizer to train/test data and save tokenized modified version
# * TODO:  should save different data processing outputs in different folders..  (raw, input, tokenized, modified..)
# 
# * sentence piece model used for tokenizing must match , otherwise our embeddings will be bad

# In[127]:


if EXTRACT_SENTPIECE_LABELED_TOKENIZED_SEQ: 
    # train
    text = pd.read_csv(TRAIN_LABELED_DATA_FILE_PATH)
    text["Sequence"] = text["Sequence"].apply(sp.encode_as_pieces).apply(" ".join)
    text.to_csv("tokenized_"+TRAIN_LABELED_DATA_FILE_PATH,index=False,encoding="utf-8")
    # test file
    text = pd.read_csv(TEST_LABELED_DATA_FILE_PATH)
    text["Sequence"] = text["Sequence"].apply(sp.encode_as_pieces).apply(" ".join)
    text.to_csv("tokenized_"+TEST_LABELED_DATA_FILE_PATH,index=False,encoding="utf-8")


# In[128]:


text.head().loc[:, :].replace(1, "__label__"+pd.Series(text.columns, text.columns))


# ## Supervised FastText model
# * can use gensim instead
# * Fasttext expect speicifc format
# * doesn't really do multilabel
# 
# *FT expects All the labels to start with the `__label__` prefix, which is how fastText recognize what is a label or what is a word.  `__label__`.
#     * code : https://stackoverflow.com/questions/37032043/how-to-replace-a-value-in-a-pandas-dataframe-with-column-name-based-on-a-conditi   + add `__label__` prefix , `text.head().loc[:, :].replace(1, "__label__"+pd.Series(text.columns, text.columns))`
#     
#     * might need to remove spaces from targets/column names to avoid leaks?? 
#     * also remove 0s otherwise can be a leak as well 
#     * write out as space delimited text file siomply

# In[129]:


# df_train_raw = pd.read_csv(TRAIN_LABELED_DATA_FILE_PATH)
# print("train shape",df_train_raw.shape)
# df_test_raw = pd.read_csv(TEST_LABELED_DATA_FILE_PATH)
# print("test shape",df_test_raw.shape)


# * make version of text data for fasttext , with suffixes on labels

# In[131]:


text = pd.read_csv("tokenized_"+TRAIN_LABELED_DATA_FILE_PATH)

#### remove whitespace from columns/labels 
text.columns = text.columns.str.replace(" ","-")
text.loc[:, :].replace(1, "__label__"+pd.Series(text.columns, text.columns),inplace=True)
text.replace(0,np.nan,inplace=True)

text.to_csv("fastText_train.txt",sep=' ', index=False, header=False,quoting=None)

## same for test


text = pd.read_csv("tokenized_"+TEST_LABELED_DATA_FILE_PATH)

#### remove whitespace from columns/labels 
text.columns = text.columns.str.replace(" ","-")
text.loc[:, :].replace(1, "__label__"+pd.Series(text.columns, text.columns),inplace=True)
text.replace(0,np.nan,inplace=True)

text.to_csv("fastText_test.txt",sep=' ', index=False, header=False,quoting=None)


# #### Train supervised fasttext model
# * output of "predict" is samples, precision , recall
#     * precision can be changed to get Prec/recall at @ top k 
#     
# * the threshholdd/probabiltiies doesn't seem to work in python+Windows?? 
#     * same for autotune of hyperparams 

# In[144]:


def print_results(N, p, r):
    print("PRETRAINING_N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


# In[141]:


## https://fasttext.cc/docs/en/supervised-tutorial.html#multi-label-classification

model = fasttext.train_supervised(input="fastText_train.txt", loss='ova')


# In[142]:


# print(model.words)
print(model.labels)


# * Let's have a look on our predictions, we want as many prediction as possible (argument -1) and we want only labels with probability higher or equal to 0.5 :

# In[143]:


model.test("fastText_test.txt", k=-1, threshold=0.5)


# In[151]:


print_results(*model.test('fastText_test.txt', k=3))


# In[ ]:




