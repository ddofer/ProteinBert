#!/usr/bin/env python

import os
import json
import pickle
from itertools import islice

import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
from pyfaidx import Faidx

import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from bert import BertModelLayer


# ----- config -----

# Data
N_SEQS_TO_READ = 100000 # None
MAX_N_SEQS_TO_USE_DUMP = 1e07
ALL_DATA = True

# Language
VOCAB_SIZE = 16000
MAX_SEQ_TOKEN_LEN = 250
MASK_OUT_FREQ = 0.2
N_SEQS_FOR_SENTENCE_PIECE_TRAINING = 1000000

# Model
MODEL = ['BERT', 'EMBEDDING'][0]

# Training
MULTI_GPU = False 
BATCH_SIZE = 64
N_EPOCHS = 10 # 1000
N_EPOCHS_PER_SAVE = 1 

# Reserved tokens
N_RESERVED_SYMBOLS = 2
PAD_TOKEN = VOCAB_SIZE - 1
MASK_TOKEN = VOCAB_SIZE - 2

# Paths
MODEL_WEIGHTS_FILE = '/home/user/Desktop/protobret_weights.h5'
SP_TRAINING_CORPUS_TXT_FILE_PATH = '/home/user/Desktop/data/sentence_piece_training_corpus_tmp.txt'
DATA_DUMP_DIR = '/home/user/Desktop/data/dataset_dump/'

if ALL_DATA:
    # wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
    INPUT_FASTA_FILE_PATH = '/home/user/Downloads/uniref90.fasta'
    # wget ftp://ftp.cs.huji.ac.il/users/nadavb/cafa4_data/protein_annotations.db.gz
    ANNOTATIONS_SQLITE_FILE_PATH = '/home/user/Downloads/protein_annotations.db'
else:
    INPUT_FASTA_FILE_PATH = '/home/user/Desktop/data/uniref90_sample.fasta'
    ANNOTATIONS_SQLITE_FILE_PATH = '/home/user/Desktop/protein_annotations_sample.db'

# ----- config -----


# Helper functions

def format_quantity(quantity):
    if quantity is None:
        return 'all'
    else:
        return str(quantity)
        
def pad_tokens(tokens):
    return np.concatenate([tokens, np.full(MAX_SEQ_TOKEN_LEN - len(tokens), PAD_TOKEN)])
    
def encode_labels(labels, label_to_index, n_labels):
    
    encoded_labels = np.zeros(n_labels, dtype = np.int32)
    
    for label in labels:
        encoded_labels[label_to_index[label]] += 1
        
    return encoded_labels
    
def get_random_mask(shape, zero_freq):
    mask = np.ones(shape, dtype = bool).flatten()
    mask[:int(zero_freq * mask.size)] = False
    np.random.shuffle(mask)
    return mask.reshape(shape)
    
def format_token_id(sp, token_id):
    if token_id == PAD_TOKEN:
        return '/'
    elif token_id == MASK_TOKEN:
        return '?'
    else:
        return sp.id_to_piece(int(token_id))

def pad_to_max_len(*strings):
    max_len = max(map(len, strings))
    return [string + (max_len - len(string)) * ' ' for string in strings]


# Workflow

def install_dependencies():
    os.system('pip install bert-for-tf2')
    os.system('pip install sentencepiece')

def setup_tensorflow():

    if MULTI_GPU:
        # tf.compat.v1.disable_eager_execution() # Commenting this enables the multi gpu to work but apparently it is slower?!
        from tensorflow.keras.utils import multi_gpu_model

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def load_data():

    print('Loading %s GO annotations...' % format_quantity(N_SEQS_TO_READ))
    cnx = sqlite3.connect(ANNOTATIONS_SQLITE_FILE_PATH)
    go_annots = pd.read_sql_query('SELECT * FROM protein_annotations' + ('' if N_SEQS_TO_READ is None else (' LIMIT %d' % N_SEQS_TO_READ)), cnx)
    print('Loaded %d GO annotations (%d columns: %s)' % (go_annots.shape + (', '.join(go_annots.columns),)))

    print('Loading Faidx (%s)...' % INPUT_FASTA_FILE_PATH)
    seqs_faidx = Faidx(INPUT_FASTA_FILE_PATH)
    print('Finished loading Faidx.')

    loaded_seqs = []
    loaded_annotations = []

    for _, (uniprot_id, raw_annotations) in tqdm(go_annots[['uniprot_name', 'complete_go_annotation_indices']].iterrows(), total = len(go_annots)):
        
        seq_fasta_id = 'UniRef90_%s' % uniprot_id.split('_')[0]
        
        try:
            seq = str(seqs_faidx.fetch(seq_fasta_id, 1, seqs_faidx.index[seq_fasta_id].rlen))
            loaded_seqs.append(seq)
            loaded_annotations.append(json.loads(raw_annotations))
        except KeyError:
            print('Sequence ID %s was not found in the FASTA file.' % seq_fasta_id)
    
    assert len(loaded_seqs) == len(loaded_annotations)
    print('Successfully loaded %d sequences with annotations (of %d).' % (len(loaded_seqs), len(go_annots)))
    return loaded_seqs, loaded_annotations
    
def load_data_with_dump():
    if N_SEQS_TO_READ <= MAX_N_SEQS_TO_USE_DUMP:
        
        dump_file_path = os.path.join(DATA_DUMP_DIR, '%d_seqs.pkl' % N_SEQS_TO_READ)
        
        if os.path.exists(dump_file_path):
        
            print('Loading data from %s...' % dump_file_path)
        
            with open(dump_file_path, 'rb') as f:
                return pickle.load(f)
        else:
            
            data = load_data()
            print('Dumping data into %s...' % dump_file_path)
            
            with open(dump_file_path, 'wb') as f:
                pickle.dump(data, f)
                
            return data
    else:
        return load_data()

def train_sp(seqs):
    
    with open(SP_TRAINING_CORPUS_TXT_FILE_PATH, 'w') as f:
        for seq in islice(seqs, N_SEQS_FOR_SENTENCE_PIECE_TRAINING):
            f.write(str(seq) + '\n')
    
    # TODO: check what --hard_vocab_limit=false actually does
    # TODO Don't forget to comment-in!!!
    #spm.SentencePieceTrainer.Train('--input=%s --model_prefix=protopiece --vocab_size=%d --hard_vocab_limit=false' % \
    #        (SP_TRAINING_CORPUS_TXT_FILE_PATH,         VOCAB_SIZE - N_RESERVED_SYMBOLS))
    sp = spm.SentencePieceProcessor()
    sp.load('protopiece.model')
    return sp
    
def create_dataset():
    
    seqs, annotations = load_data_with_dump()

    sp = train_sp(seqs)
    seq_tokens = list(map(sp.encode_as_ids, seqs))
    padded_seq_tokens = np.array([pad_tokens(tokens) for tokens in seq_tokens if len(tokens) <= MAX_SEQ_TOKEN_LEN])
    print('Padded %d of %d sequences in appropriate length.' % (len(padded_seq_tokens), len(seq_tokens)))
    print('Shape of padded sequence tokens: %dx%d' % padded_seq_tokens.shape)

    unique_annotations = sorted(set.union(*map(set, annotations)))
    n_unique_annotations = len(unique_annotations)
    print('There are %d unique annotations.' % n_unique_annotations)
    annotation_to_index = {annotation: i for i, annotation in enumerate(unique_annotations)}
    encoded_annotations = np.array([encode_labels(record_annotations, annotation_to_index, n_unique_annotations) for record_annotations, record_tokens in \
            zip(annotations, seq_tokens) if len(record_tokens) <= MAX_SEQ_TOKEN_LEN])
    print('Shape of encoded annotations: %dx%d' % encoded_annotations.shape)
    
    return sp, unique_annotations, padded_seq_tokens, encoded_annotations
    
def create_bert_model(annotation_dim):
    
    # TODO: Use keras-bert transformer library..
    
    input_seq_ids = keras.layers.Input(shape = (MAX_SEQ_TOKEN_LEN,), dtype = np.int32)
    input_annotations = keras.layers.Input(shape = (annotation_dim,))

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
    
    l_bert.trainable = True
    
    bert_output = l_bert(input_seq_ids) # Shape: Batch x MAX_SEQ_TOKEN_LEN x 768
    # Shape change: Batch x annotation_dim --> Batch x MAX_SEQ_TOKEN_LEN x annotation_dim
    repeated_annotations = keras.layers.RepeatVector(MAX_SEQ_TOKEN_LEN)(input_annotations) 
    bert_output_with_annotations = keras.layers.Concatenate()([bert_output, repeated_annotations]) # Shape: Batch x MAX_SEQ_TOKEN_LEN x (768 + annotation_dim)
    hidden = keras.layers.Dense(1024, activation = 'tanh', name = 'token_guess')(bert_output_with_annotations)
    
    output_seqs = keras.layers.Dense(VOCAB_SIZE, activation = 'softmax', name = 'output_seqs')(hidden)
    output_annotations_per_position = keras.layers.Dense(annotation_dim, activation = 'sigmoid', name = 'output_annotations_per_position')(hidden)
    output_annotations = K.mean(output_annotations_per_position, axis = 1)
    
    model = keras.Model(inputs = [input_seq_ids, input_annotations], outputs = [output_seqs, output_annotations])
    return model
    
def create_embedding_model():
    # XXX Delete me
    pass

def create_model(annotation_dim):
    if MODEL == 'BERT':
        return create_bert_model(annotation_dim)
    elif MODEL == 'EMBEDDING':
        return create_embedding_model()
        
def compile_model(model):

    if MULTI_GPU:
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus = 4)
    
    optimizer = keras.optimizers.Adam(lr = 1e-06, amsgrad = True)
    model.compile(optimizer = optimizer, loss = ['sparse_categorical_crossentropy', 'binary_crossentropy'])
    model.summary()
    
def create_and_compile_model(annotation_dim):
    model = create_model(annotation_dim)
    compile_model(model)
    return model

def fit_model(model, encoded_tokens, encoded_annotations):
    for i in range(N_EPOCHS):

        print('Epoch %d:' % (i + 1))
        
        # Mask out the tokens.
        seq_mask = get_random_mask(encoded_tokens.shape, MASK_OUT_FREQ)
        masked_tokens = np.where(seq_mask, encoded_tokens, MASK_TOKEN)
        
        # Mask out the annotations.
        annotation_mask = get_random_mask(encoded_annotations.shape, MASK_OUT_FREQ)
        masked_annotations = np.where(annotation_mask, encoded_annotations, 0)
        
        model.fit([masked_tokens, masked_annotations], [encoded_tokens, encoded_annotations], batch_size = BATCH_SIZE)
        
        if i % N_EPOCHS_PER_SAVE == 0:
            model.save_weights(MODEL_WEIGHTS_FILE)
            
def display_model_result(model, encoded_tokens, encoded_annotations, i = 0):
    
    record_encoded_tokens = encoded_tokens[i, :]
    record_ecnoded_annotations = encoded_annotations[i, :]
    
    record_seq_mask = get_random_mask(record_encoded_tokens.shape, MASK_OUT_FREQ)
    record_masked_tokens = np.where(record_seq_mask, record_encoded_tokens, MASK_TOKEN)
    
    record_annotation_mask = get_random_mask(record_ecnoded_annotations.shape, MASK_OUT_FREQ)
    record_masked_annotations = np.where(record_annotation_mask, record_ecnoded_annotations, 0)

    # TODO: Need to refactor it
    
    predicted_token_ids = model.predict([masked_totken_ids.reshape(1, -1)])[0, :, :].argmax(axis = -1)
        
    original_formatted_tokens = []
    predicted_formatted_tokens = []
    
    for original_token_id, mask_bit, predicted_token_id in zip(original_token_ids, used_mask, predicted_token_ids):
        mask_surrounding = '' if mask_bit else '?'
        original_formatted_token, predicted_formatted_token = pad_to_max_len(mask_surrounding +                 format_token_id(original_token_id) + mask_surrounding, format_token_id(predicted_token_id))
        original_formatted_tokens.append(original_formatted_token)
        predicted_formatted_tokens.append(predicted_formatted_token)
        
    print(' '.join(original_formatted_tokens) + '\n' + ' '.join(predicted_formatted_tokens))
    
def run_pipeline():
    
    #create_bert_model(291) # XXX
    install_dependencies()
    setup_tensorflow()
            
    sp, unique_annotations, encoded_tokens, encoded_annotations = create_dataset()
    model = create_and_compile_model(len(unique_annotations))
    fit_model(model, encoded_tokens, encoded_annotations)
    
    display_model_result(i = 13)

if __name__ == '__main__':
    run_pipeline()
