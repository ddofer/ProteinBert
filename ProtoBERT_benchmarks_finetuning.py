import os
import pickle
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing


# from tensorflow import keras 
from tensorflow import keras as keras
from IPython.display import display

# from pwas.shared_utils.util import log

from sklearn.metrics import matthews_corrcoef, r2_score, f1_score, precision_score, recall_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, roc_auc_score, log_loss # DAN
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


KERAS_SEQ_VEC_SCRIPT_PATH = "keras_seq_vec_transformer.py"
# ""/cs/phd/nadavb/github_projects/keras_seq_vec_transformer/keras_seq_vec_transformer.py" # ORIGINAL path in CSE server
with open(KERAS_SEQ_VEC_SCRIPT_PATH, 'r') as f:
    '''
    Some horrible hacks. To make keras_seq_vec_transformer work with tf.keras instead of just keras we run the following instead of
    just exec(f.read()).
    Note that also above we run from tensorflow import keras instead of just import keras
    '''
    import tensorflow.keras.backend as K 
    from tensorflow.keras.layers import LayerNormalization
    exec('\n'.join([line for line in f.read().splitlines() if not line.startswith('import keras') and not \
            line.startswith('from keras_layer_normalization')]))

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
N_ANNOTATIONS = 8943

# BENCHMARKS_DIR = '/cs/phd/nadavb/cafa_project/data/proteomic_benchmarks'
BENCHMARKS_DIR = '../data'

PRETRAINED_MODEL_WEIGHTS_FILE_PATH = '../models/epoch_28530_sample_91400000.pkl'

BENCHMARKS = [
    'signalP_binary.dataset',
    'scop.dataset',
    'fluorescence', 
    'secondary_structure',
    
#     'remote_homology', ## currently has bug
#     'disorder_secondary_structure', ## error : ValueError: could not convert string to float: '<PAD>' 
#     "PhosphositePTM.dataset", ## error : ValueError: could not convert string to float: '<PAD>'  ### 9 min per epoch
    "stability" 
             ]
  #### "phosphoserine.dataset",   ## sequence texts and label lengths are different - dataset needs to be changed in advance or dropped

MAX_GLOBAL_SEQ_LEN = 450#450 # 450 # set it here to make it easier to change. should be length used in training.  # DAN

MAX_ALLOWED_INPUT_SEQ = MAX_GLOBAL_SEQ_LEN - 2
EARLY_STOPPING_PATIENCE=5

MAX_EPOCHS=50 #80 # max train epochs,
BATCH_SIZE = 16#16

DEBUG_MODE = False
USE_PRETRAINED_WEIGHTS = True

FAST_RUN = True # True # False#
FAST_SAMPLE_RATIO = 0.2 # if doing fast run, downsample to roughly this percent of data from train data

if FAST_RUN:
    MAX_EPOCHS = 8
#     BATCH_SIZE = BATCH_SIZE//2 
#     MAX_ALLOWED_INPUT_SEQ = 132
    EARLY_STOPPING_PATIENCE=1

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
ADDED_TOKENS_PER_SEQ = 2
n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + len(ALL_AAS) for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)

def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
#     return [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in seq] + [additional_token_to_index['<END>']] # ORIG
    return [aa_to_token_index.get(aa, other_token_index) for aa in seq]


def tokenize_seqs(seqs,max_seq_len=MAX_GLOBAL_SEQ_LEN):   
    tokenized_seqs = additional_token_to_index['<PAD>'] * np.ones((len(seqs), max_seq_len))
    for i, seq in enumerate(seqs):
        tokenized_seq = tokenize_seq(seq)
        assert len(tokenized_seq) <= max_seq_len
        tokenized_seqs[i, :len(tokenized_seq)] = tokenized_seq
    return tokenized_seqs

def create_model(max_seq_len):   
    input_seq_layer = keras.layers.Input(shape = (max_seq_len,), dtype = np.int32, name = 'input-seq')
    input_annoatations_layer = keras.layers.Input(shape = (N_ANNOTATIONS,), dtype = np.float32, name = 'input-annotations')
    output_seq_layer, output_annoatations_layer = TransformerAutoEncoder(vocab_size = n_tokens, d_vec = N_ANNOTATIONS, output_vec_activation = 'sigmoid', name = 'auto-encoder')([input_seq_layer, input_annoatations_layer])
    output_seq_layer = keras.layers.Reshape(output_seq_layer.shape[1:], name = 'output_seq_layer')(output_seq_layer)
    output_annoatations_layer = keras.layers.Reshape(output_annoatations_layer.shape[1:], name = 'output_annoatations_layer')(output_annoatations_layer)
    return keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = [output_seq_layer, output_annoatations_layer])

def load_model_weights(model, path):
    with open(path, 'rb') as f:
        model_weights, optimizer_weights = pickle.load(f)
        model.set_weights(model_weights)

def evaluate(Y_pred, raw_y_true, is_y_numeric, is_y_seq, unique_labels):     
    n_labels = len(unique_labels)
    Y_pred_classes = Y_pred.argmax(axis = -1) 
#     if n_labels<20:
    if len(np.unique(Y_pred_classes))<20:
        try:
            print("classification Report\n")
            print(classification_report(raw_y_true,Y_pred_classes))
        except:pass    
    try:
        print("MCC %.4f%" % matthews_corrcoef(raw_y_true,Y_pred_classes))
    except:pass
    try:
        print("F1 - macro avg %.4f%" % f1_score(raw_y_true,Y_pred_classes, average='macro'))
        print("precision - micro avg %.2f%%" % (100 * precision_score(raw_y_true,Y_pred_classes, average='micro')))
        print("Recall - macro avg %.2f%%" % (100 * recall_score(raw_y_true,Y_pred_classes, average='micro')))              
        print("balanced_accuracy_score %.4f%" % balanced_accuracy_score(raw_y_true,Y_pred_classes))
    except:pass    
    try:
        print("r2 %.4f%" % r2_score(raw_y_true,Y_pred.flatten())) # doesn't work? DAN
        print("mean_absolute_error %.4f%" % mean_absolute_error(raw_y_true,Y_pred.flatten()))
    except: pass   
    try: 
        print("roc_auc_score %.4f%" % roc_auc_score(raw_y_true,Y_pred[:,1]))
        print("log_loss %.4f%" % log_loss(raw_y_true,Y_pred[:,1]))
    except:pass    
    if is_y_numeric:
        results = pd.DataFrame({'true': raw_y_true, 'pred': Y_pred.flatten()})
        print("spearman's rho (correlation)",results.corr(method="spearman"))
        print('R^2 score: %.2g' % r2_score(results['true'], results['pred']))        
        print("mean absolute error score %.4g" % mean_absolute_error(results['true'], results['pred']))                    
    else:
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
        confusion_matrix = results.groupby(['true', 'pred']).size().unstack().fillna(0)
        if DEBUG_MODE:
    #         if len(set(unique_labels))<20 and False: # ORIG ? why False?
            if len(set(unique_labels))<20:
                print('Confusion matrix:')
                display(confusion_matrix)

            #         accuracy = (results['true'].astype(int) == results['pred'].astype(int)).mean() ## DAN - added .astype(int)  -causes error due to pad output
        accuracy = (results['true'] == results['pred']).mean() # currently broken
    
        imbalance = (results['true'].value_counts().max() / len(results))
        print('Accuracy: %.2f%%' % (100 * accuracy))
        print('Imbalance (most common label): %.2f%%' % (100 * imbalance))
        if len(set(unique_labels)) == 2:
            y_true = results['true'].astype(float)
            y_pred = results['pred'].astype(float)
            print('MCC: %.2f%%' % (100 * matthews_corrcoef(y_true, y_pred)))           
            print("F1 - macro avg %.2f%%" % (100 * f1_score(y_true, y_pred, average='macro')))       
            print("precision - micro avg %.2f%%" % (100 * precision_score(y_true, y_pred, average='micro')))
            print("Recall - macro avg %.2f%%" % (100 * recall_score(y_true, y_pred, average='micro')))

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
#     if "remote_homology" in benchmark_name:
#         train_set = pd.concat([train_set.sample(frac=FAST_SAMPLE_RATIO,random_state=42),train_set.drop_duplicates('labels')]).drop_duplicates()
    if is_y_discrete:
        print("discrete_sampling")
#         t1,_ = train_test_split(train_set,stratify=train_set['labels'],train_size=FAST_SAMPLE_RATIO,random_state=42) # stratified sampling 0 for multiclass
        t1,_ = train_test_split(train_set,train_size=FAST_SAMPLE_RATIO,random_state=42) # random, unstratified sampling - avoids errors with to ofew smaples per class (we combine it with at least 1 example per class below)
        t2 = train_set.drop_duplicates('labels')
#         t2 = train_set.groupby(['labels']).apply(lambda x: x.sample(2))
        print("t2 shape",t2.shape)
        train_set = pd.concat([t1,t2]).drop_duplicates()
                
    else:
        print("continous sampling")
        train_set = train_set.sample(frac=FAST_SAMPLE_RATIO,random_state=42)
    valid_set = valid_set.sample(frac=FAST_SAMPLE_RATIO)
    test_set = test_set.sample(frac=FAST_SAMPLE_RATIO)   
    return train_set, valid_set, test_set


def get_y_type(y):
    '''
    Determining which of the following y is:
    1. Numeric (could be either probabalistic (i.e. in the range 0-1) or not)
    2. Sequence
    3. Categorical ### confusing - we return is_y_probability ??? # DAN
    '''
#     is_y_numeric = np.issubdtype(y.dtype, np.floating) ##ORIG
    is_y_numeric = np.issubdtype(y.dtype, np.number)
    if is_y_numeric:
#         is_y_probability =  not isinstance(y, int)# ORIG - #y.min() >= 0 and y.max() <= 1
#         is_y_probability =  (y.min() >= 0 and y.max() <= 1)  #ORIG
        is_y_probability =  (not(y% 1  == 0).all())
    ## another way to check: (df[col] % 1  == 0).all() ## https://stackoverflow.com/questions/49249860/how-to-check-if-float-pandas-column-contains-only-integer-numbers
        is_y_seq = False
    else: 
        is_y_probability = False
        is_y_seq = y.astype(str).str.len().max()>14  # duuuuuudeeeeee
        print('Numeric (%sprobabilistic) label' % ('' if is_y_probability else 'not '))
    is_y_discrete = (not (is_y_seq or is_y_probability)) # may want to add condition based on cardinality? Otherwise fails for fluorescence
    return is_y_numeric, is_y_seq, is_y_probability,is_y_discrete

## v1
# def build_fine_tuning_model(is_y_numeric, is_y_probability, is_y_seq,max_seq_len, n_labels,learning_rate= 1e-04):
#     model = create_model(max_seq_len)
#     if USE_PRETRAINED_WEIGHTS:
#         load_model_weights(model, PRETRAINED_MODEL_WEIGHTS_FILE_PATH)
#     input_seq_layer, input_annoatations_layer = model.input
#     output_seq_layer, output_annoatations_layer = model.output    
#     if is_y_numeric:
#         output_layer = keras.layers.Dense(1, activation = ('sigmoid' if is_y_probability else None))(output_annoatations_layer)
#         loss = 'binary_crossentropy' if (not is_y_probability) else 'mse' # DAN fix
#     elif is_y_seq:
#         output_layer = keras.layers.Dense(n_labels + 1, activation = 'softmax')(output_seq_layer)
#         loss = 'categorical_crossentropy'
    
#     else: # non-seq categorical
#         print("build_fine_tuning_model - n_labels",n_labels)
# #         if n_labels ==2:
# #             loss = 'binary_crossentropy' 
# #         else:
#         loss = 'categorical_crossentropy' 
#         output_layer = keras.layers.Dense(n_labels, activation = 'softmax')(output_annoatations_layer)
    
#     if n_labels ==2:
#         loss = 'binary_crossentropy' 
#         output_layer = keras.layers.Dense(1, activation = ('sigmoid'))(output_annoatations_layer) # NEW
    
#     if is_y_probability:
#         output_layer = keras.layers.Dense(1, activation = ('sigmoid' if is_y_probability else None))(output_annoatations_layer)
#         loss = 'mse'
        
# #     if DEBUG_MODE: print("model loss:",loss)
#     print("\n model-Loss:",loss)
#     model = keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = output_layer)
#     model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = loss) 
#     return model


## v2
def build_fine_tuning_model(is_y_numeric, is_y_probability, is_y_seq,max_seq_len, n_labels,learning_rate= 1e-04):
    model = create_model(max_seq_len)
    if USE_PRETRAINED_WEIGHTS:
        load_model_weights(model, PRETRAINED_MODEL_WEIGHTS_FILE_PATH)
    input_seq_layer, input_annoatations_layer = model.input
    output_seq_layer, output_annoatations_layer = model.output    
    
    if is_y_seq:
        output_layer = keras.layers.Dense(n_labels + 1, activation = 'softmax')(output_seq_layer)
        loss = 'categorical_crossentropy'
        
    else:
        if is_y_probability:
            output_layer = keras.layers.Dense(1, activation = None)(output_annoatations_layer)
            loss = 'mse'
        else:
            if n_labels ==2:
                loss = 'binary_crossentropy'
                output_layer = keras.layers.Dense(1, activation = 'sigmoid')(output_annoatations_layer)
            elif n_labels>2:
                loss = 'categorical_crossentropy' 
                output_layer = keras.layers.Dense(n_labels, activation = 'softmax')(output_annoatations_layer)
            else: print("ERROR")
        
#     if DEBUG_MODE: print("model loss:",loss)
    print("\n model-Loss:",loss)
    print("\n n_labels:",n_labels)
    model = keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = output_layer)
    model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = loss) 
    return model

def encode_seq_Y(raw_Y, max_seq_len, n_labels, label_to_index):
    # +1 for padding
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
    else: # non-seq categorical   - turn into binary matrix    
        #### i replaced our encode_categorical_y with TF's to_categoircal . TF's expects an integer input. 
        ### tODO - double check that order of classes/integer is kept! should be fine due to use of sklearn to i
        
#         train_Y = encode_categorical_y(train_set['labels'], n_labels, label_to_index)
#         valid_Y = encode_categorical_y(valid_set['labels'], n_labels, label_to_index)
        train_Y = to_categorical(train_set['labels'],num_classes=n_labels)
        valid_Y = to_categorical(valid_set['labels'],num_classes=n_labels)
        
    return train_X, valid_X, test_X, train_Y, valid_Y

def train_and_eval(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq,
                   n_labels, unique_labels, label_to_index):
    train_X, valid_X, test_X, train_Y, valid_Y = preproc_benchmark_dataset(train_set, valid_set, test_set, 
                                                                           n_labels, label_to_index, MAX_GLOBAL_SEQ_LEN, is_y_numeric, is_y_seq)
    model = build_fine_tuning_model(is_y_numeric, is_y_probability, is_y_seq, MAX_GLOBAL_SEQ_LEN, n_labels)
    ### Train model, with early stopping on validation set
    model.fit(train_X, train_Y,
              batch_size = BATCH_SIZE,
              validation_data=(valid_X,valid_Y),
              callbacks = [ReduceLROnPlateau(patience=2,factor=0.35), EarlyStopping(patience=EARLY_STOPPING_PATIENCE)],
              epochs = MAX_EPOCHS,
              validation_batch_size=BATCH_SIZE,
              verbose=1)
    
    #### DAN: NOTE - keras already keeps evaluation, train error data.
    print('\n*** Training-set performance: ***')
    train_Y_pred = model.predict(train_X)
    evaluate(train_Y_pred, train_set['labels'].values, is_y_numeric, is_y_seq, unique_labels)
    if DEBUG_MODE:
        print('*** Validation-set performance: ***')
        valid_Y_pred = model.predict(valid_X)
        evaluate(valid_Y_pred, valid_set['labels'].values, is_y_numeric, is_y_seq, unique_labels)
    print('*** Test-set performance: ***')
    test_Y_pred = model.predict(test_X)
    evaluate(test_Y_pred, test_set['labels'].values, is_y_numeric, is_y_seq, unique_labels)    
    print('\n' * 3)
    
def train_and_eval_after_removing_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq,
                                                n_labels, unique_labels, label_to_index):
    filtered_train_set = train_set[train_set['text'].str.len() <= MAX_ALLOWED_INPUT_SEQ]
    filtered_valid_set = valid_set[valid_set['text'].str.len() <= MAX_ALLOWED_INPUT_SEQ]
    filtered_test_set = test_set[test_set['text'].str.len() <= MAX_ALLOWED_INPUT_SEQ]
    n_removed_train_set = len(train_set) - len(filtered_train_set)
    ptg_removed_train_set = 100 * n_removed_train_set / len(train_set)
    n_removed_valid_set = len(valid_set) - len(filtered_valid_set)
    ptg_removed_valid_set = 100 * n_removed_valid_set / len(valid_set)
    n_removed_test_set = len(test_set) - len(filtered_test_set)
    ptg_removed_test_set = 100 * n_removed_test_set / len(test_set)
    print('Trying to remove too long sequences. Removed %d of %d (%.1g%%) of the training set, %d of %d (%.1g%%) of the validation set and %d of %d (%.1g%%) of the test set' %
            (n_removed_train_set, len(train_set), ptg_removed_train_set, n_removed_valid_set, len(valid_set), ptg_removed_valid_set, n_removed_test_set, len(test_set), ptg_removed_test_set))
    train_and_eval(filtered_train_set, filtered_valid_set, filtered_test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)
    
def truncate_dataset(dataset, is_y_seq):
    if is_y_seq:
        return chunk_df(dataset, MAX_ALLOWED_INPUT_SEQ-2) ## -2 - avoid edge case of length equal to max +1 : would only leave the padding tokenss
    else:
        dataset = dataset.copy()
        dataset['text'] = dataset['text'].apply(lambda seq: seq[:MAX_ALLOWED_INPUT_SEQ])
        return dataset
    
def train_and_eval_after_trancating_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability,
                                                  is_y_seq, n_labels, unique_labels, label_to_index):
    print('Will now truncate too-long sequences.')
    train_and_eval(truncate_dataset(train_set, is_y_seq), truncate_dataset(valid_set, is_y_seq), truncate_dataset(test_set, is_y_seq), is_y_numeric, is_y_probability, is_y_seq,
                   n_labels, unique_labels, label_to_index)

def run_benchmark(benchmark_name):
    
    print('========== %s ==========' % benchmark_name)   
    print('\n')
    text_file_path, train_set, valid_set, test_set = load_benchmark_dataset(benchmark_name)
    is_y_numeric, is_y_seq, is_y_probability,is_y_discrete = get_y_type(train_set['labels'])
       
        ## fast hack, the regression/continous targets are being marked as discrete accidentally , this makes them get downsampled properly
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
        
        
    if FAST_RUN: train_set, valid_set, test_set = fast_run(train_set, valid_set, test_set,is_y_discrete) # dan change - use new is_y_discrete
    print(f'{len(train_set)} training-set records, {len(valid_set)} valid-set records, {len(test_set)} test-set records')     
    
#   ## add start and end tokens in advance of processing 
    train_set["text"] = ['<START>']+train_set["text"]+['<END>']
    valid_set["text"] = ['<START>']+valid_set["text"]+['<END>']
    test_set["text"] = ['<START>']+test_set["text"]+['<END>']
    
    if DEBUG_MODE:
        print(train_set.dtypes)
        print(valid_set.dtypes)

    
    if is_y_numeric:
        ### stupid ugly hack - the numeric/continous targets don't have n_labels/unique_labels, but their surroudning functions expect them - I hack in a default for now, expect a refactor - DAN
        n_labels=2 # default hack
        unique_labels=("0","1") # default hack
        label_to_index = {}
        print("y numeric label hack done")

    if is_y_discrete: ## new
        n_labels = train_set['labels'].nunique()
    
    if not is_y_numeric:
        if DEBUG_MODE: print("\n middle is_y_numeric preproc entered")
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
        #       ##i.e y is discrete classes
                train_set['labels'] = le.fit_transform(train_set['labels'])
                print("le",le)
                print("le.classes_",le.classes_)
                print(len(list(le.classes_)))
                valid_set['labels'] = le.transform(valid_set['labels'])
                test_set['labels'] = le.transform(test_set['labels'])

    if DEBUG_MODE:
        print("n_labels",n_labels)
     
    if max(train_set["text"].str.len().max(), valid_set["text"].str.len().max(), test_set["text"].str.len().max()) <= MAX_ALLOWED_INPUT_SEQ:
        train_and_eval(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)
    else:
        ### stupid hack until 2d array error length fixed:
        if ((benchmark_name =='secondary_structure') or (benchmark_name =='remote_homology')) :  ### stupid hack
            train_and_eval_after_removing_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)
        
        else: ### stupid hack
            ## in 2d structure: error: ValueError: arrays must all be same length
            train_and_eval_after_trancating_too_long_seqs(train_set, valid_set, test_set, is_y_numeric, is_y_probability, is_y_seq, n_labels, unique_labels, label_to_index)

            
for benchmark_name in BENCHMARKS:
    run_benchmark(benchmark_name)



# # text_file_path, train_set, valid_set, test_set = load_benchmark_dataset('remote_homology')
# # text_file_path, train_set, valid_set, test_set = load_benchmark_dataset('signalP_binary.dataset')

# text_file_path, train_set, valid_set, test_set = load_benchmark_dataset('fluorescence')

# is_y_numeric, is_y_seq, is_y_probability,is_y_discrete = get_y_type(train_set['labels'])
# print("dtype labels",train_set['labels'].dtype)
# train_set['labels'] = pd.to_numeric(train_set['labels'],downcast="integer")
# print("downcast dtype labels",train_set['labels'].dtype)
# print(train_set['labels'].head())
# display(train_set['labels'].describe())

# print("train # unique labels",train_set.labels.nunique())
# print("valid # unique labels",valid_set.labels.nunique())
# print("test # unique labels",test_set.labels.nunique())
# print(test_set['labels'].head())


