#!/usr/bin/env python
# coding: utf-8


import os
import h5py
import keras
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.utils import to_categorical
# from pwas.shared_utils.util import log
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from sklearn.metrics import matthews_corrcoef, r2_score, f1_score, precision_score, recall_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error


import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)

from training_utils import *
from config import *


def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
    return [aa_to_token_index.get(aa, other_token_index) for aa in seq]

def tokenize_seqs(seqs,max_seq_len=MAX_GLOBAL_SEQ_LEN):   
    tokenized_seqs = additional_token_to_index['<PAD>'] * np.ones((len(seqs), max_seq_len))
    for i, seq in enumerate(seqs):
        tokenized_seq = tokenize_seq(seq)
        assert len(tokenized_seq) <= max_seq_len
        tokenized_seqs[i, :len(tokenized_seq)] = tokenized_seq
    return tokenized_seqs

def evaluate(Y_pred, raw_y_true, is_y_numeric, is_y_seq, unique_labels):     
    n_labels = len(unique_labels)
    Y_pred_classes = Y_pred.argmax(axis = -1) 
    if len(np.unique(Y_pred_classes))<20:
        try:
            print("classification Report\n")
            print(classification_report(raw_y_true,Y_pred_classes)) 
        except:pass
    try:
        if n_labels <=2:
            print("roc_auc_score %.4f" % roc_auc_score(raw_y_true,Y_pred))
            print("log_loss %.4f" % log_loss(raw_y_true,Y_pred))
            print("MCC %.4f" % matthews_corrcoef(raw_y_true,Y_pred_classes))
        if n_labels >2:
            try: 
                print("multiclass roc_auc_score %.4f" % roc_auc_score(raw_y_true,Y_pred)) ## Y_pred[:,1]
                print("log_loss %.4f" % log_loss(raw_y_true,Y_pred)) ## [:,1]
            except:pass  
    except:pass
    try:
        print("F1 - micro avg %.4f" % f1_score(raw_y_true,Y_pred_classes, average='micro'))
        print("precision - micro avg %.2f" % (100 * precision_score(raw_y_true,Y_pred_classes, average='micro')))
        print("Recall - macro avg %.2f" % (100 * recall_score(raw_y_true,Y_pred_classes, average='micro')))              
        print("balanced_accuracy_score %.4f" % balanced_accuracy_score(raw_y_true,Y_pred_classes))
    except:pass    
    try:
        print("r2 %.4f" % r2_score(raw_y_true,Y_pred.flatten())) # doesn't work? DAN
        print("mean_absolute_error %.4f" % mean_absolute_error(raw_y_true,Y_pred.flatten()))
    except: pass
    if is_y_numeric:
        try:
            results = pd.DataFrame({'true': raw_y_true, 'pred': Y_pred.flatten()})
            print("spearman's rho (correlation)",results.corr(method="spearman"))
            print('R^2 score: %.2g' % r2_score(results['true'], results['pred']))        
            print("mean absolute error score %.4g" % mean_absolute_error(results['true'], results['pred']))  
        except:
            print("evaluation of is_y_numeric failed")
            pass
    else:
        try:
            if is_y_seq:
                results_true = []
                results_pred = []
                for true_seq, pred_seq in zip(raw_y_true, Y_pred.argmax(axis = -1)):
                    for true_token, pred_token_index in zip(true_seq, pred_seq):
                        results_true.append(true_token)
                        results_pred.append('<PAD>' if pred_token_index == n_labels else unique_labels[pred_token_index])
                results = pd.DataFrame({'true': results_true, 'pred': results_pred})
            else:
                predicted_labels = [unique_labels[i] for i in Y_pred.argmax(axis = -1)]
                results = pd.DataFrame({'true': raw_y_true, 'pred': predicted_labels})
        except: pass
        try:
            confusion_matrix = results.groupby(['true', 'pred']).size().unstack().fillna(0)
            if FINETUNING_DEBUG_MODE:
                if len(set(unique_labels))<20:
                    print('Confusion matrix:')
                    display(confusion_matrix)
            accuracy = (results['true'] == results['pred']).mean()
            imbalance = (results['true'].value_counts().max() / len(results))
            print('Accuracy: %.2f' % (100 * accuracy))
            print('Imbalance (most common label): %.2f' % (100 * imbalance))
        except: pass    
        if len(set(unique_labels)) == 2:
            try:
                y_true = results['true'].astype(float)
                y_pred = results['pred'].astype(float)
                print('MCC: %.2f' % (100 * matthews_corrcoef(y_true, y_pred)))           
                print("F1 - macro avg %.2f" % (100 * f1_score(y_true, y_pred, average='macro')))       
                print("precision - micro avg %.2f" % (100 * precision_score(y_true, y_pred, average='micro')))
                print("Recall - macro avg %.2f" % (100 * recall_score(y_true, y_pred, average='micro')))
            except: pass 

def chunk_string(string:str, chunk_size:int):
    return [string[i:(chunk_size + i)] for i in range(0, len(string), chunk_size)]

def chunk_df(df, chunk_size:int):
    return pd.DataFrame({col: df[col].apply(chunk_string, chunk_size = chunk_size).explode().values for col in df.columns})

def load_benchmark_dataset(benchmark_name): 
    text_file_path = os.path.join(BENCHMARKS_DIR, '%s.benchmark.txt' % benchmark_name)    
    train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.train.csv' % benchmark_name)
    valid_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.valid.csv' % benchmark_name)
    test_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.test.csv' % benchmark_name)
    train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
    if os.path.exists(valid_set_file_path):
        valid_set = pd.read_csv(valid_set_file_path).dropna().drop_duplicates()
    else:
        print(f"validation set {valid_set_file_path} missing")
        print("splitting train to train and random validation set")
        try: 
            train_set, valid_set = train_test_split(train_set, stratify=train_set['labels'],test_size=0.1,random_state=42)
            print("Stratified sampling of validation set")
        except:
            print("randomly sampling validation set")
            train_set, valid_set = train_test_split(train_set,test_size=0.1,random_state=42)
    test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()
    return text_file_path, train_set, valid_set, test_set

def fast_run(train_set, valid_set, test_set, is_y_discrete):
    if is_y_discrete:
        print("discrete_sampling")
        t1,_ = train_test_split(train_set, train_size=FINETUNING_FAST_SAMPLE_RATIO, random_state=42)
        t2 = train_set.drop_duplicates('labels')
        train_set = pd.concat([t1,t2]).drop_duplicates()
    else:
        print("continous sampling")
        train_set = train_set.sample(frac=FINETUNING_FAST_SAMPLE_RATIO, random_state=42)
    valid_set = valid_set.sample(frac=FINETUNING_FAST_SAMPLE_RATIO)
    return train_set, valid_set, test_set

def get_y_type(y):
    is_y_numeric = np.issubdtype(y.dtype, np.number)
    if is_y_numeric:
        is_y_probability =  (not(y% 1  == 0).all())
        is_y_seq = False
    else: 
        is_y_probability = False
        is_y_seq = y.astype(str).str.len().max() > 14  # duuuuuudeeeeee
        print('Numeric (%sprobabilistic) label' % ('' if is_y_probability else 'not '))
    is_y_discrete = (not (is_y_seq or is_y_probability))
    return is_y_numeric, is_y_seq, is_y_probability,is_y_discrete

def build_fine_tuning_model(is_y_numeric, is_y_probability, is_y_seq,max_seq_len, n_labels,learning_rate= 3e-04):
    model = MODEL_TYPE.create_model(max_seq_len, vocab_size=n_tokens, n_annotations = N_ANNOTATIONS)
    if FINETUNING_USE_PRETRAINED_WEIGHTS:
        if os.path.exists(FINETUNING_PRETRAINED_MODEL_WEIGHTS_FILE_PATH):
            load_model(model, FINETUNING_PRETRAINED_MODEL_WEIGHTS_FILE_PATH)
        else:
            print('Cannot load weights file %s..' % (FINETUNING_PRETRAINED_MODEL_WEIGHTS_FILE_PATH))
    input_seq_layer, input_annoatations_layer = model.input
    output_seq_layer, output_annoatations_layer = model.output
    print(output_annoatations_layer)
    if is_y_seq:
        output_layer = keras.layers.Dense(n_labels + 1, activation = 'softmax', dtype='float32')(output_seq_layer)
        loss = 'categorical_crossentropy'
    else:
        if is_y_probability:
            output_layer = keras.layers.Dense(1, activation = None)(output_annoatations_layer)
            loss = 'mse'
        else:
            if n_labels ==2:
                loss = 'binary_crossentropy'
                output_layer = keras.layers.Dense(1, activation = 'sigmoid', dtype='float32')(output_annoatations_layer)
            elif n_labels>2:
                loss = 'categorical_crossentropy' 
                output_layer = keras.layers.Dense(n_labels, activation = 'softmax', dtype='float32')(output_annoatations_layer)
            else: print("ERROR")
    print("model loss:",loss)
    model = keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = output_layer)
    model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = loss) 
    return model

def encode_seq_Y(raw_Y, max_seq_len, n_labels, label_to_index):
    Y = np.zeros((len(raw_Y), max_seq_len, n_labels + 1), dtype = np.int8)
    for i, seq in enumerate(raw_Y):
        for j, token in enumerate(seq):
            Y[i, j, label_to_index[token]] = 1
        Y[i, np.arange(len(seq), max_seq_len), n_labels] = 1
    return Y

def preproc_benchmark_dataset(train_set, valid_set, test_set, n_labels, label_to_index, max_seq_len, is_y_numeric, is_y_seq):
    train_X = [
        tokenize_seqs(train_set['text'].values, max_seq_len).astype(np.int32),
        np.zeros((len(train_set), N_ANNOTATIONS), dtype = np.int8)
    ]    
    valid_X = [
        tokenize_seqs(valid_set['text'].values, max_seq_len).astype(np.int32),
        np.zeros((len(valid_set), N_ANNOTATIONS), dtype = np.int8)
    ]    
    test_X = [
        tokenize_seqs(test_set['text'].values, max_seq_len).astype(np.int32),
        np.zeros((len(test_set), N_ANNOTATIONS), dtype = np.int8)
    ]    
    if is_y_numeric:
        train_Y = train_set['labels'].values
        valid_Y = valid_set['labels'].values
    elif is_y_seq:  
        print("y-seq y encoding")
        train_Y = encode_seq_Y(train_set['labels'], max_seq_len, n_labels, label_to_index)
        valid_Y = encode_seq_Y(valid_set['labels'], max_seq_len, n_labels, label_to_index)
    else:
        train_Y = to_categorical(train_set['labels'],num_classes=n_labels)
        valid_Y = to_categorical(valid_set['labels'],num_classes=n_labels)
        if FINETUNING_DEBUG_MODE:
            print(train_Y)
            print("train_Y unique values:",len(np.unique(train_Y)))
            print("valid_Y unique values:",len(np.unique(valid_Y)))
    return train_X, valid_X, test_X, train_Y, valid_Y

def train_and_eval(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index):
    train_X, valid_X, test_X, train_Y, valid_Y = preproc_benchmark_dataset(train_set, valid_set, test_set, n_labels, label_to_index, MAX_GLOBAL_SEQ_LEN, is_y_numeric, is_y_seq)
    model = build_fine_tuning_model(is_y_numeric, is_y_probability, is_y_seq, MAX_GLOBAL_SEQ_LEN, n_labels)
    model.fit(train_X, train_Y,
              batch_size = FINETUNING_BATCH_SIZE,
              validation_data=(valid_X,valid_Y),
              callbacks = [ReduceLROnPlateau(patience=2,factor=0.35), EarlyStopping(patience=FINETUNING_EARLY_STOPPING_PATIENCE)],
              epochs = FINETUNING_MAX_EPOCHS,
              # validation_batch_size=FINETUNING_BATCH_SIZE,
              verbose=1)
    print('\n*** Training-set performance: ***')
    train_Y_pred = model.predict(train_X)
    evaluate(train_Y_pred, train_set['labels'].values, is_y_numeric, is_y_seq, unique_labels)
    if FINETUNING_DEBUG_MODE:
        print('*** Validation-set performance: ***')
        valid_Y_pred = model.predict(valid_X)
        evaluate(valid_Y_pred, valid_set['labels'].values, is_y_numeric, is_y_seq, unique_labels)
    print('*** Test-set performance: ***')
    test_Y_pred = model.predict(test_X)
    evaluate(test_Y_pred, test_set['labels'].values, is_y_numeric, is_y_seq, unique_labels)    
    print('\n' * 3)
    
def train_and_eval_after_removing_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index):
    filtered_train_set = train_set[train_set['text'].str.len() <= FINETUNING_MAX_ALLOWED_INPUT_SEQ]
    filtered_valid_set = valid_set[valid_set['text'].str.len() <= FINETUNING_MAX_ALLOWED_INPUT_SEQ]
    filtered_test_set = test_set[test_set['text'].str.len() <= FINETUNING_MAX_ALLOWED_INPUT_SEQ]
    n_removed_train_set = len(train_set) - len(filtered_train_set)
    ptg_removed_train_set = 100 * n_removed_train_set / len(train_set)
    n_removed_valid_set = len(valid_set) - len(filtered_valid_set)
    ptg_removed_valid_set = 100 * n_removed_valid_set / len(valid_set)
    n_removed_test_set = len(test_set) - len(filtered_test_set)
    ptg_removed_test_set = 100 * n_removed_test_set / len(test_set)
    print('Trying to remove too long sequences. Removed %d of %d (%.1g%%) of the training set, %d of %d (%.1g%%) of the validation set and %d of %d (%.1g%%) of the test set' % (n_removed_train_set, len(train_set), ptg_removed_train_set, n_removed_valid_set, len(valid_set), ptg_removed_valid_set, n_removed_test_set, len(test_set), ptg_removed_test_set))
    train_and_eval(filtered_train_set, filtered_valid_set, filtered_test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)
    
def truncate_dataset(dataset, is_y_seq):
    if is_y_seq:
        return chunk_df(dataset, FINETUNING_MAX_ALLOWED_INPUT_SEQ - 2)
    else:
        dataset = dataset.copy()
        dataset['text'] = dataset['text'].apply(lambda seq: seq[:FINETUNING_MAX_ALLOWED_INPUT_SEQ])
        return dataset
    
def train_and_eval_after_trancating_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index):
    print('Will now truncate too-long sequences.')
    train_and_eval(truncate_dataset(train_set, is_y_seq), truncate_dataset(valid_set, is_y_seq), truncate_dataset(test_set, is_y_seq), is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)

def run_benchmark(benchmark_name):
    print('========== %s ==========' % benchmark_name)   
    print('\n')
    text_file_path, train_set, valid_set, test_set = load_benchmark_dataset(benchmark_name)
    is_y_numeric, is_y_seq, is_y_probability,is_y_discrete = get_y_type(train_set['labels'])
    if ((benchmark_name =='fluorescence') or (benchmark_name =="stability")):
        is_y_numeric=True
        is_y_discrete=False
        is_y_probability=True
    else:
        is_y_discrete=True
    if (benchmark_name =='secondary_structure'):
        is_y_discrete=False
        is_y_probability=False
        is_y_seq=True
    if is_y_numeric: print("y_numeric")
    if is_y_seq: print("y_seq")
    if is_y_probability: print("y_probability")
    if is_y_discrete: print("y_discrete")
    if FINETUNING_FAST_RUN: train_set, valid_set, test_set = fast_run(train_set, valid_set, test_set, is_y_discrete) # dan change - use new is_y_discrete
    print(f'{len(train_set)} training-set records, {len(valid_set)} valid-set records, {len(test_set)} test-set records')
    train_set["text"] = ['<START>']+train_set["text"]+['<END>']
    valid_set["text"] = ['<START>']+valid_set["text"]+['<END>']
    test_set["text"] = ['<START>']+test_set["text"]+['<END>']
    if FINETUNING_DEBUG_MODE:
        print(train_set.dtypes)
    if is_y_probability:
        n_labels=2
        unique_labels=("0","1")
        label_to_index = {}
        print("y numeric label hack done")
    if is_y_discrete:
        n_labels = train_set['labels'].nunique()
        unique_labels = sorted(train_set['labels'].unique())
        n_labels = len(unique_labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
    if (not is_y_numeric):
        if FINETUNING_DEBUG_MODE: print("\n middle is_y_numeric preproc entered")
        train_set['labels'] = train_set['labels'].astype(str)
        if is_y_seq:            
            unique_labels = sorted(set.union(*train_set['labels'].apply(set))) 
        else:
            unique_labels = sorted(train_set['labels'].unique())                               
        n_labels = len(unique_labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}        
        print('Sequence output with %d tokens.' % n_labels if is_y_seq else 'Categorical output with %d labels.' % n_labels)
    if (not is_y_numeric):
        if not is_y_seq:
            if is_y_discrete:
                le = preprocessing.LabelEncoder()
                print("\n label encoding class labels")
                train_set['labels'] = le.fit_transform(train_set['labels'])
                print("le.classes_",le.classes_)
                print(len(list(le.classes_)))
                valid_set['labels'] = le.transform(valid_set['labels'])
                test_set['labels'] = le.transform(test_set['labels'])
    if (benchmark_name =='remote_homology'): is_y_numeric=False
    if FINETUNING_DEBUG_MODE:
        print("n_labels",n_labels)
        try: 
            print("train targets:")
            print(train_set['labels'].value_counts(normalize=True))
        except: pass
    if max(train_set["text"].str.len().max(), valid_set["text"].str.len().max(), test_set["text"].str.len().max()) <= FINETUNING_MAX_ALLOWED_INPUT_SEQ:
        train_and_eval(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)
    else:
        if (benchmark_name =='secondary_structure') :
            train_and_eval_after_removing_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)
        else:
            train_and_eval_after_trancating_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)

for benchmark_name in FINETUNING_BENCHMARKS:
    run_benchmark(benchmark_name)
