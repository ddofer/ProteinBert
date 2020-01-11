#!/usr/bin/env python
# coding: utf-8

'''
import gzip
from Bio import SeqIO
import re
REP_ID_PATTERN = re.compile(r'RepID=(\S+)')
def parse_rep_id(raw_desc):
    rep_id, =   REP_ID_PATTERN.findall(raw_desc)
    return rep_id

with gzip.open('data/uniref90.fasta.gz', 'rt') as fi, open('data/uniref90.fasta', 'w') as fo:
    for r in SeqIO.parse(fi, 'fasta'):
        #r.id = parse_rep_id(r.description)
        SeqIO.write(r, fo, 'fasta')

print('done')
raise Exception()
'''

import os
import bert
import gzip
import json
import h5py
import random
import sqlite3
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import tensorflow as tf
import sentencepiece as spm
from tensorflow import keras
import matplotlib.pyplot as plt
from bert import BertModelLayer
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import RepeatVector, Permute, Embedding

# !wget ftp://ftp.cs.huji.ac.il/users/nadavb/cafa_4_partial_dataset/protein_tokens.h5 -O ~/protein_tokens.h5
# !wget ftp://ftp.cs.huji.ac.il/users/nadavb/cafa_4_partial_dataset/protopiece.model -O ~/protopiece.model

# ----- config -----

MODEL = ['BERT', 'EMBEDDING'][0]
MULTI_GPU = False
VOCAB_SIZE = 16000
N_RESERVED_SYMBOLS = 2
PAD_TOKEN = VOCAB_SIZE - 1
MASK_TOKEN = VOCAB_SIZE - 2
MAX_LEN = 250
SAMPLES_SLICE = slice(1000) # slice(None)
N_EPOCHS = 1000
N_EPOCHS_PER_SAVE = 1
EPOCH_SIZE = None
BATCH_SIZE = 64
WEIGHTS_FILE = os.path.expanduser('/home/user/Desktop/protobret_weights.h5')
DATASET_H5_FILE_PATH = os.path.expanduser('/home/user/Desktop/protein_tokens.h5')
MASK_OUT_FREQ = 0.2
N_SEQS = 1000000
INPUT_FASTA_FILE_PATH = '/home/user/Desktop/data/uniref90.fasta.gz'
CORPUS_TXT_FILE_PATH = '/home/user/Desktop/data/sentence_test.txt'

# ----- config -----

if MULTI_GPU:
    # tf.compat.v1.disable_eager_execution() # Commenting this enables the multi gpu to work but apparently it is slower?!
    from tensorflow.keras.utils import multi_gpu_model


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def install_pkgs():
    os.system('pip install bert-for-tf2')
    os.system('pip install sentencepiece')

def load_data():
    # cnx = sqlite3.connect('db.sqlite')
    # with gzip.open('/home/user/Desktop/data/protein_annotations.db.gz') as f:
        # cnx = sqlite3.connect('db.sqlite')
        # cnx = sqlite3.connect('/home/user/Desktop/data/protein_annotations.db.gz')
        # df = pd.read_sql_query("SELECT * FROM table_name", cnx)
    cnx = sqlite3.connect('protein_annotations.db')
    go_annots = pd.read_sql_query("SELECT * FROM protein_annotations LIMIT 10000", cnx)
    # print(go_annots.shape)
    # print(go_annots.columns)
    #print(go_annots['uniprot_name'])
    #print(go_annots['complete_go_annotation_indices'])
    # print(go_annots.iloc[:3])
    from pyfaidx import Fasta
    seqs = Fasta('/home/user/Desktop/data/uniref90.fasta')
    # print(len(seqs.keys()), list(seqs.keys())[:10])
    # print(seqs['UniRef90_Q6GZX1'])
    # print(go_annots.sample())
    X, Y = [], []
    vocab = 'MNAKYDTDQGVGRMLFLGTIGLAVVVGGLMAYGYYYDGKTPSSGTSFHTASPSFSSRYRY'
    worked = 0
    for _, (uniprot_id, y) in tqdm(go_annots[['uniprot_name', 'complete_go_annotation_indices']].iterrows(), total=len(go_annots)):
        try:
            key = 'UniRef90_%s' % uniprot_id.split('_')[0]
            x = str(seqs[key])
            # x = seq
            # y = go_annots.loc[id]['complete_go_annotation_indices']
            worked+=1
        except:
            x = ''.join([random.choice(vocab) for i in range(len(vocab))])
        X.append(x)
        Y.append(json.loads(y))
    return X, Y

def encode(X):
    """
    with gzip.open(INPUT_FASTA_FILE_PATH, 'rt') as input_fasta_file, open(CORPUS_TXT_FILE_PATH, 'w') as output_txt_file:
        for i, record in enumerate(tqdm(SeqIO.parse(input_fasta_file, 'fasta'), total=N_SEQS)):
            if N_SEQS is not None and i >= N_SEQS: break
            output_txt_file.write(str(record.seq) + '\n')
    """
    with open(CORPUS_TXT_FILE_PATH, 'w') as output_txt_file:
        for i, record in enumerate(tqdm(X)):
            if N_SEQS is not None and i >= N_SEQS: break
            output_txt_file.write(str(record) + '\n')

    spm.SentencePieceTrainer.Train('--input=%s --model_prefix=protopiece --vocab_size=%d' % (CORPUS_TXT_FILE_PATH, VOCAB_SIZE - N_RESERVED_SYMBOLS))
    sp = spm.SentencePieceProcessor()
    sp.load('protopiece.model')

    # example_seq = 'MRYTVLIALQGALLLLLLIDDGQGQSPYPYPGMPCNSSRQCGLGTCVHSRCAHCSSDGTLCSPEDPTMVWPCCPESSCQLVVG' + 'LPSLVNHYNCLPNQCTDSSQCPGGFGCMTRRSKCELCKADGEACNSPYLDWRKDKECCSGYCHTEARGLEGVCIDPKKIFCTP' + 'KNPWQLAPYPPSYHQPTTLRPPTSLYDSWLMSGFLVKSTTAPSTQEEEDDY'
    # print(sp.encode_as_pieces(example_seq))
    # print(sp.encode_as_ids(example_seq))

    return sp

def pad_tokens(tokens):
    return np.concatenate([tokens, PAD_TOKEN * np.ones(MAX_LEN - len(tokens))])


seqs, y = load_data()
print(len(seqs), len(y))
print(seqs[:10], y[:10])
sp = encode(seqs)
seq_tokens = list(map(sp.encode_as_ids, seqs))
padded_seq_tokens = np.array([pad_tokens(tokens) for tokens in seq_tokens if len(tokens) <= MAX_LEN])
print('%d of %d sequences are of right length.' % (len(seq_tokens), len(padded_seq_tokens)))
print(padded_seq_tokens.shape)


y_padded = np.array([pad_tokens(tokens) for tokens in y])


if MODEL == 'BERT':
    l_input_ids = keras.layers.Input(shape=(MAX_LEN,), dtype=np.int32)
    # l_token_type_ids = keras.layers.Input(shape=(MAX_LEN,), dtype=np.int32)

    l_bert = BertModelLayer(**BertModelLayer.Params(

        # embedding params
        vocab_size               = VOCAB_SIZE,
        use_token_type           = True,
        use_position_embeddings  = True,
        token_type_vocab_size    = 2,

        # transformer encoder params
        num_heads                = 4,
        num_layers               = 4,
        hidden_size              = 768,
        hidden_dropout           = 0.1,
        intermediate_size        = 2 * 512,
        intermediate_activation  = "gelu",

        # see arXiv:1902.00751 (adapter-BERT)
        adapter_size             = None,

        # True for ALBERT (arXiv:1909.11942)
        shared_layer             = True,
        # None for BERT, wordpiece embedding size for ALBERT
        embedding_size           = None,

        # any other Keras layer params
        name                     = "bert",
    ))
    bert_output = l_bert(l_input_ids)

    # bert_output = l_bert([l_input_ids, l_token_type_ids])
    token_guess_output = keras.layers.Dense(VOCAB_SIZE, activation='softmax', name='token_guess')(bert_output)
    l_bert.trainable = True
    # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=bert_output)
    # model.build(input_shape=[(None, MAX_LEN), (None, MAX_LEN)])
    model = keras.Model(inputs=[l_input_ids], outputs=bert_output)
    model.build(input_shape=[(None, MAX_LEN)])


elif MODEL == 'EMBEDDING':
    l_input_ids = keras.layers.Input(shape = (MAX_LEN,), dtype = np.int32)
    l_input_ids2 = Embedding(VOCAB_SIZE, VOCAB_SIZE)(l_input_ids)
    token_guess_output = l_input_ids2
    model = keras.Model(inputs = l_input_ids, outputs = token_guess_output)
    model.build(input_shape = (None, MAX_LEN))



print(model.summary())

optimizer = keras.optimizers.Adam(lr = 1e-06, amsgrad = True)
if MULTI_GPU: gpu_model = multi_gpu_model(model, gpus=4)
else: gpu_model = model
gpu_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')



# fit

for i in range(N_EPOCHS):

    print('Epoch %d:' % (i + 1))
    mask = np.ones_like(padded_seq_tokens, dtype=bool).flatten()
    mask[:int(MASK_OUT_FREQ * mask.size)] = False
    np.random.shuffle(mask)
    mask = mask.reshape(padded_seq_tokens.shape)
    masked_dataset_tokens = np.where(mask, padded_seq_tokens, MASK_TOKEN)
    print(masked_dataset_tokens)

    if EPOCH_SIZE is None: epoch_mask = np.arange(len(padded_seq_tokens))
    else: epoch_mask = np.random.randint(0, len(padded_seq_tokens), EPOCH_SIZE)
    epoch_X = masked_dataset_tokens[epoch_mask, :]
    epoch_Y = padded_seq_tokens[epoch_mask, :]
    # epoch_label = y_padded[epoch_mask, :]
    print(epoch_X.shape)
    print(epoch_Y.shape)
    # print(epoch_label.shape)
    gpu_model.fit(epoch_X, epoch_Y, batch_size = BATCH_SIZE)
    if i % N_EPOCHS_PER_SAVE == 0:
        gpu_model.save_weights(WEIGHTS_FILE)


# predict

# sp = spm.SentencePieceProcessor()
# sp.load('protopiece.model')

def format_token_id(token_id):
    if token_id == PAD_TOKEN:
        return '/'
    elif token_id == MASK_TOKEN:
        return '?'
    else:
        return sp.id_to_piece(int(token_id))

def pad_to_max_len(*strings):
    max_len = max(map(len, strings))
    return [string + (max_len - len(string)) * ' ' for string in strings]

def display_model_result(i = 0):
    
    original_token_ids = padded_seq_tokens[i, :]
    used_mask = mask[i, :]
    masked_totken_ids = masked_dataset_tokens[i, :]
    predicted_token_ids = model.predict(masked_totken_ids.reshape(1, -1))[0, :, :].argmax(axis = -1)
    
#     print(np.concatenate([original_token_ids.reshape(-1, 1), masked_totken_ids.reshape(-1, 1), \
#             predicted_token_ids.reshape(-1, 1)], axis = 1))
    
    original_formatted_tokens = []
    predicted_formatted_tokens = []
    
    for original_token_id, mask_bit, predicted_token_id in zip(original_token_ids, used_mask, predicted_token_ids):
        mask_surrounding = '' if mask_bit else '?'
        original_formatted_token, predicted_formatted_token = pad_to_max_len(mask_surrounding +                 format_token_id(original_token_id) + mask_surrounding, format_token_id(predicted_token_id))
        original_formatted_tokens.append(original_formatted_token)
        predicted_formatted_tokens.append(predicted_formatted_token)
        
    print(' '.join(original_formatted_tokens) + '\n' + ' '.join(predicted_formatted_tokens))

display_model_result(i = 13)

