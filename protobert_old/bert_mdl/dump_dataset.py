#!/usr/bin/env python

import os
import gc
import json
import pickle
import traceback
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
# from bert_weaponized import *
import os
import re
import sqlite3
import json

import numpy as np
import pandas as pd
import matplotlib as plt

from Bio import SeqIO

from IPython.display import display


# ----- config -----

# Data
N_SEQS_TO_READ = 10
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
N_EPOCHS = 1 #  10000
N_EPOCHS_PER_SAVE = 1 

# Reserved tokens
N_RESERVED_SYMBOLS = 2
PAD_TOKEN = VOCAB_SIZE - 1
MASK_TOKEN = VOCAB_SIZE - 2

# Paths
DATA_DIR = '/home/user/Desktop/data/'
MODEL_WEIGHTS_FILE = '/home/user/Desktop/protobret_weights.h5'
SP_TRAINING_CORPUS_TXT_FILE_PATH = '/home/user/Desktop/data/sentence_piece_training_corpus_tmp.txt'
DATA_DUMP_DIR = '/home/user/Desktop/data/dataset_dump/'

if ALL_DATA:
    # wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
    INPUT_FASTA_FILE_PATH = '/home/user/Downloads/uniref90.fasta'
    # INPUT_FASTA_FILE_PATH = os.path.join(DATA_DIR, 'uniref90.fasta')

    # wget ftp://ftp.cs.huji.ac.il/users/nadavb/cafa4_data/protein_annotations.db.gz
    # wget ftp://ftp.cs.huji.ac.il/users/nadavb/cafa4_data/target_seqs.csv
    ANNOTATIONS_SQLITE_FILE_PATH = os.path.join(DATA_DIR, 'protein_annotations.db')
    # ANNOTATIONS_SQLITE_FILE_PATH = '/home/user/Downloads/protein_annotations.db'
else:
    INPUT_FASTA_FILE_PATH = os.path.join(DATA_DIR, 'uniref90_sample.fasta')
    # INPUT_FASTA_FILE_PATH = '/home/user/Desktop/data/uniref90_sample.fasta'

    ANNOTATIONS_SQLITE_FILE_PATH = os.path.join(DATA_DIR, 'protein_annotations_sample.db')
    # ANNOTATIONS_SQLITE_FILE_PATH = '/home/user/Desktop/protein_annotations_sample.db'

GO_ANNOTATIONS = os.path.join(DATA_DIR, 'go_annotations.csv')
TARGET_SEQS = os.path.join(DATA_DIR, 'target_seqs.csv')

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

def load_data():

    print('Loading %s GO annotations...' % format_quantity(N_SEQS_TO_READ))
    cnx = sqlite3.connect(ANNOTATIONS_SQLITE_FILE_PATH)
    go_annots = pd.read_sql_query('SELECT * FROM protein_annotations' + ('' if N_SEQS_TO_READ is None else (' ORDER BY RANDOM() LIMIT %d' % N_SEQS_TO_READ)), cnx)
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
    

def train_sp(seqs):
    
    with open(SP_TRAINING_CORPUS_TXT_FILE_PATH, 'w') as f:
        for seq in islice(seqs, N_SEQS_FOR_SENTENCE_PIECE_TRAINING):
            f.write(str(seq) + '\n')

    spm.SentencePieceTrainer.Train('--input=%s --model_prefix=protopiece --vocab_size=%d --hard_vocab_limit=false' % \
            (SP_TRAINING_CORPUS_TXT_FILE_PATH,         VOCAB_SIZE - N_RESERVED_SYMBOLS))
    sp = spm.SentencePieceProcessor()
    sp.load('protopiece.model')
    return sp

def create_dataset(seqs=None, annotations=None):

    if seqs is None or annotations is None:
        seqs, annotations = load_data()
    try:
        np.save('seqs.npy', np.array(seqs))
        np.save('annotations.npy', np.array(annotations))
    except:
        import traceback
        traceback.print_exc()

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


def create_submission_files(prediction_model, output_dir, target_seqs, go_annotations_meta, model_id=1):
    '''
    prediction model is expected to be a function that takes two arguments (seq and annotations) and returns the final,
    refined annotations. The protein seq is expected as a simple string of aa letters. The input annotations is expected
    as a list of integers, as provided in the SQLITE DB. The output annotations is expected as a dictionary, mapping each
    annotation integer (the same indices as in the input) into a confidence score between 0 to 1.

    See: https://www.biofunctionprediction.org/cafa-targets/CAFA4_rules%20_01_2020_v4.pdf
    '''

    MAX_ANNOTATIONS = 1500

    OUTPUT_FILE_NAME_PATTERN = 'linialgroup_%d_%s_go.txt'
    SUBMISSION_FILE_PREFIX = 'AUTHOR Linial' + '\n' + ('MODEL %d' % model_id) + '\n' + 'KEYWORDS machine learning.'

    go_annotation_index_to_id = go_annotations_meta['id']

    for idx, (tax_id, tax_target_seqs) in tqdm(enumerate(target_seqs.groupby('taxa_id'))):
        gc.collect()
        print('Preparing submission for tax ID %s...' % tax_id)
        formatted_predictions = []

        for _, (cafa_id, seq, raw_go_annotation_indices) in tqdm(tax_target_seqs[
            ['cafa_id', 'seq', 'complete_go_annotation_indices']].iterrows(), total=len(tax_target_seqs), desc='Tax id: %s (%s/%s)' % (tax_id, idx, len(target_seqs.groupby('taxa_id')))):
            go_annotation_indices = [] if pd.isnull(raw_go_annotation_indices) else json.loads(
                raw_go_annotation_indices)
            annotation_predictions = prediction_model(seq, go_annotation_indices)
            top_predicted_annotations = list(sorted(annotation_predictions.items(), reverse=True))[:MAX_ANNOTATIONS]
            top_predicted_annotations = [(annotation_index, round(score, 2)) for annotation_index, score in
                                         top_predicted_annotations]
            formatted_predictions.extend(
                ['%s\t%s\t%s' % (cafa_id, go_annotation_index_to_id[annotation_index], score) for
                 annotation_index, score in top_predicted_annotations if score > 0])

        tax_submission_content = SUBMISSION_FILE_PREFIX + '\n' + '\n'.join(formatted_predictions) + '\n' + 'END'

        with open(os.path.join(output_dir, OUTPUT_FILE_NAME_PATTERN % (model_id, tax_id)), 'w') as f:
            f.write(tax_submission_content)
        del formatted_predictions
        gc.collect()

    print('Done.')

def run_pipeline():

    seqs, annotations = load_data()
    sp, unique_annotations, encoded_tokens, encoded_annotations = create_dataset(seqs, annotations)


if __name__ == '__main__':
    run_pipeline()
