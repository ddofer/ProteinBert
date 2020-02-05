#!/usr/bin/env python

import os
import gc
import json
import pickle
import traceback
from pmap import p_map
from itertools import islice

import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm, trange
from pyfaidx import Faidx

import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
# from bert import BertModelLayer
from bert_weaponized import *
import os
import re
import sqlite3
import json

import numpy as np
import pandas as pd
import matplotlib as plt

from Bio import SeqIO

from IPython.display import display
TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

# ----- config -----

# Data
N_SEQS_TO_READ = 100000 # None
MAX_N_SEQS_TO_USE_DUMP = 1e07
ALL_DATA = True

# Language
VOCAB_SIZE = 1000 # 16000
MAX_SEQ_TOKEN_LEN = 250
MASK_OUT_FREQ = 0.2
N_SEQS_FOR_SENTENCE_PIECE_TRAINING = 100

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
        
def pad_tokens(tokens, max_len = MAX_SEQ_TOKEN_LEN):
    return np.concatenate([tokens, np.full(max_len - len(tokens), PAD_TOKEN)])
    
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

def load_all_data():
    print('Loading all Seqs..')
    seqs = pickle.load(open('/home/user/Desktop/seqs_all.pkl', 'rb'))
    print('All Seqs loaded!. Len: ', len(seqs))
    print('Loading all Annotations..')
    annotations = pickle.load(open('/home/user/Desktop/annots_all.pkl', 'rb'))
    print('All Annotations loaded!. Len: ', len(annotations))
    return seqs, annotations

def train_sp(seqs):
    print('Saving Corpus..')
    with open(SP_TRAINING_CORPUS_TXT_FILE_PATH, 'w') as f:
        for seq in islice(seqs, N_SEQS_FOR_SENTENCE_PIECE_TRAINING):
            f.write(str(seq) + '\n')
    print('Corpus Saved!')
    print('Training SentencePiece...')
    # TODO: check what --hard_vocab_limit=false actually does
    # TODO Don't forget to comment-in!!!
    spm.SentencePieceTrainer.Train('--input=%s --model_prefix=protopiece --vocab_size=%d --hard_vocab_limit=false' % \
            (SP_TRAINING_CORPUS_TXT_FILE_PATH,         VOCAB_SIZE - N_RESERVED_SYMBOLS))
    print('SentencePiece Training Done!')
    sp = spm.SentencePieceProcessor()
    sp.load('protopiece.model')
    return sp

def encode_data():
    seqs, annotations = load_all_data()

    #sp = train_sp(seqs)
    print('Encoding seqs..')
    ALL_AAS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ALL_AAS += [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
    aa_to_index = {aa: i for i, aa in enumerate(ALL_AAS)}
    #seq_tokens = list(map(sp.encode_as_ids, tqdm(seqs)))
    # seq_tokens = [[aa_to_index[aa] for aa in seq] for seq in tqdm(seqs)]
    for i, seq in tqdm(enumerate(seqs)):
        seqs[i] = [aa_to_index[aa] for aa in seq]

    return seqs, annotations

def create_dataset():
    
    global VOCAB_SIZE

    annotations = pickle.load(open('/home/user/Desktop/annots_all.pkl', 'rb'))

    data_prefix = '/home/user/Desktop/cafa/dump/seqs_'

    max_idx = int(max([float(i.replace('seqs_', '').replace('.pkl', '')) for i in os.listdir('/home/user/Desktop/cafa/dump/') if 'seqs' in i]))
    print('Max idx: ', max_idx)
    b_size = 100000
    max_idx = b_size * 2
    seq_tokens = []
    for i in trange(0, max_idx, b_size):
        seq_tokens += pickle.load(open(data_prefix + str(i) + '.pkl', 'rb'))

    # seqs, annotations = encode_data()

    ALL_AAS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ALL_AAS += [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
    VOCAB_SIZE = len(ALL_AAS)

    # TODO We now need to update vocab_size to len(ALL_AAS)
    # seq_tokens = list(p_map(sp.encode_as_ids, seqs, cpus=2, verbose=True))
    padded_seq_tokens = np.array([pad_tokens(tokens) for tokens in tqdm(seq_tokens) if len(tokens) <= MAX_SEQ_TOKEN_LEN], dtype='int8')
    print('Padded %d of %d sequences in appropriate length.' % (len(padded_seq_tokens), len(seq_tokens)))
    print('Shape of padded sequence tokens: %dx%d' % padded_seq_tokens.shape)

    # unique_annotations = sorted(set.union(*map(set, tqdm(annotations, desc='Unique Annots'))))
    # pickle.dump(unique_annotations, open('unique_a.pkl', 'wb'))

    unique_annotations = pickle.load(open('unique_a.pkl', 'rb'))

    n_unique_annotations = len(unique_annotations)
    print('There are %d unique annotations.' % n_unique_annotations)
    annotation_to_index = {annotation: i for i, annotation in enumerate(unique_annotations)}
    encoded_annotations = np.array([encode_labels(record_annotations, annotation_to_index, n_unique_annotations) for record_annotations, record_tokens in zip(annotations, seq_tokens) if len(record_tokens) <= MAX_SEQ_TOKEN_LEN], dtype='int8')
    print('Shape of encoded annotations: %dx%d' % encoded_annotations.shape)
    sp = None
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
        # hidden_size              = 768,
        hidden_size              = 32,
        hidden_dropout           = 0.1,
        # intermediate_size        = 2 * 512,
        intermediate_size      = 32,
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

    output_annotations = keras.layers.Lambda(lambda x: x, name="output_annotations")(output_annotations)

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

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def binary_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def compile_model(model):

    if MULTI_GPU:
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus = 4)
    
    optimizer = keras.optimizers.Adam(lr = 1e-06, amsgrad = True)
    model.compile(optimizer = optimizer, loss = ['sparse_categorical_crossentropy', 'binary_crossentropy'], metrics={'output_annotations': binary_auc})
    model.summary()
    return model

def create_and_compile_model(annotation_dim):
    model = create_model(annotation_dim)
    model = compile_model(model)
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
        print(encoded_annotations[0])
        print(encoded_tokens[0])
        model.fit([masked_tokens, masked_annotations], [encoded_tokens, encoded_annotations], batch_size = BATCH_SIZE )
        
        if i % N_EPOCHS_PER_SAVE == 0:
            model.save_weights(MODEL_WEIGHTS_FILE)
    return model


def process_submission_batch(prediction_model, formatted_predictions, batch_cafa_ids, batch_seqs, batch_go_annotation_indices, go_annotation_indices):
    MAX_ANNOTATIONS_PER_PROTEIN = 1500

    batch_annotation_predictions = prediction_model(batch_seqs, batch_go_annotation_indices)

    for cafa_id, annotation_predictions in zip(batch_cafa_ids, batch_annotation_predictions):
        top_predicted_annotations = list(sorted(annotation_predictions.items(), reverse=True)) \
            [:MAX_ANNOTATIONS_PER_PROTEIN]
        top_predicted_annotations = [(annotation_index, round(score, 2)) for annotation_index, score in \
                                     top_predicted_annotations]
        formatted_predictions.extend(['%s\t%s\t%s' % (cafa_id, go_annotation_indices[annotation_index], score) for \
                                      annotation_index, score in top_predicted_annotations if score > 0])


def create_submission_files(prediction_model, batch_size, output_dir, target_seqs, go_annotation_indices, model_id=1):
    '''
    prediction model is expected to be a function that takes two arguments (seqs and annotations) and returns the final,
    refined annotations. The function is expected to work in batch (i.e. receive multiple inputs and produce multiple
    outputs). The protein seqs is expected as a list of strings of aa letters. The input annotations are expected
    as a list of list of integers, as provided in the SQLITE DB. There should be full correspondence between each seq
    string to each set of annotations (list of integers); each pair is considered a distinct protein. The output
    annotations is expected as a list of dictionaries, mapping each annotation integer (the same indices provided by
    the SQLITE DB) into a confidence score between 0 to 1. Each output dictionary corresponds to the corresponding input
    protein.

    See: https://www.biofunctionprediction.org/cafa-targets/CAFA4_rules%20_01_2020_v4.pdf
    '''

    OUTPUT_FILE_NAME_PATTERN = 'linialgroup_%d_%s_go.txt'
    SUBMISSION_FILE_PREFIX = 'AUTHOR Linial' + '\n' + ('MODEL %d' % model_id) + '\n' + 'KEYWORDS machine learning.'

    for tax_id, tax_target_seqs in target_seqs.groupby('taxa_id'):

        print('Preparing submission for tax ID %s...' % tax_id)
        formatted_predictions = []
        batch_cafa_ids = []
        batch_seqs = []
        batch_go_annotation_indices = []

        for _, (cafa_id, seq, raw_go_annotation_indices) in tax_target_seqs[['cafa_id', 'seq', 'complete_go_annotation_indices']].iterrows():

            batch_cafa_ids.append(cafa_id)
            batch_seqs.append(seq)
            batch_go_annotation_indices.append([] if pd.isnull(raw_go_annotation_indices) else \
                                                   json.loads(raw_go_annotation_indices))

            if len(batch_cafa_ids) >= batch_size:
                process_submission_batch(prediction_model, formatted_predictions, batch_cafa_ids, batch_seqs, batch_go_annotation_indices, go_annotation_indices)
                batch_cafa_ids = []
                batch_seqs = []
                batch_go_annotation_indices = []

        if len(batch_cafa_ids) > 0:
            process_submission_batch(prediction_model, formatted_predictions, batch_cafa_ids, batch_seqs, batch_go_annotation_indices, go_annotation_indices)

        tax_submission_content = SUBMISSION_FILE_PREFIX + '\n' + '\n'.join(formatted_predictions) + '\n' + 'END'

        with open(os.path.join(output_dir, OUTPUT_FILE_NAME_PATTERN % (model_id, tax_id)), 'w') as f:
            f.write(tax_submission_content)

    print('Done.')


def create_prediction_model(sp, model, unique_annotations):
    
    ALL_AAS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ALL_AAS += [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
    aa_to_index = {aa: i for i, aa in enumerate(ALL_AAS)}
    
    annotation_to_index = {annotation: i for i, annotation in enumerate(unique_annotations)}

    def prediction_model(batch_seqs, batch_go_annotation_indices):

        max_len = max(map(len, batch_seqs))

        batch_padded_seq_tokens = np.array([pad_tokens([aa_to_index[aa] for aa in seq], max_len) for seq in batch_seqs])
        batch_encoded_annotations = np.array([encode_labels(go_annotation_indices, annotation_to_index, len(unique_annotations)) for \
                go_annotation_indices in batch_go_annotation_indices])
                
        # _, batch_pred_annotation_scores = model.predict(batch_padded_seq_tokens, batch_encoded_annotations)
        token_input = batch_padded_seq_tokens
        seg_input = np.zeros((token_input.shape[0], token_input.shape[1]))
        mask_input = np.ones((token_input.shape[0], token_input.shape[1]))

        _, batch_pred_annotation_scores = model.predict([batch_padded_seq_tokens, seg_input, mask_input])
        batch_pred_annotation_scores_as_dicts = [{unique_annotations[i]: score  for i, score in enumerate(pred_annotation_scores)} for \
                pred_annotation_scores in batch_pred_annotation_scores]
        return batch_pred_annotation_scores_as_dicts

        # seq_token = sp.encode_as_ids(seq)
        
        #else:
        #    return {go_annotation_index: 1.0 for go_annotation_index in go_annotation_indices}

    return prediction_model


def run_pipeline():
    # install_dependencies()
    setup_tensorflow()
    sp, unique_annotations, encoded_tokens, encoded_annotations = create_dataset()
    # model = create_and_compile_model(len(unique_annotations))
    # model = fit_model(model, encoded_tokens, encoded_annotations)
    ALL_AAS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ALL_AAS += [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
    aa_to_index = {aa: i for i, aa in enumerate(ALL_AAS)}

    model, token_dict = unsupervised_weaponized_bert(encoded_tokens, aa_to_index, encoded_annotations, MAX_SEQ_TOKEN_LEN, PAD_TOKEN)
    pred = exract_embeddings(model, token_dict, encoded_tokens, encoded_annotations)
    print(pred.shape)
    model = train_weaponized_bert(model, token_dict, encoded_tokens, encoded_annotations)

    targets = pd.read_csv(TARGET_SEQS, index_col=0)
    go_annots = pd.read_csv(GO_ANNOTATIONS, index_col='index')

    create_submission_files(create_prediction_model(sp, model, unique_annotations), 32, '.', targets ,go_annots, model_id=1)


if __name__ == '__main__':
    run_pipeline()
