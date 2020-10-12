import os
import pickle

import numpy as np
import pandas as pd

from IPython.display import display

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

from pwas.shared_utils.util import log

from tokenization import n_tokens, additional_token_to_index, tokenize_seq

BENCHMARKS_DIR = '/cs/phd/nadavb/cafa_project/data/proteomic_benchmarks'

N_ANNOTATIONS = 8943

BASELINE_SETTINGS = {
    'seq_len': 450, # TODO: Temporary hack, so we don't have to deal with long sequences for now (and all the <START>/<END>/<PAD> tokenization).
    'batch_size': 32,
    'initial_lr': 5e-05,
    'training_callbacks': [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.5),
        keras.callbacks.EarlyStopping(patience = 2),
    ],
    'max_epochs': 80,
    'max_dataset_size': None,
}

DEBUG_SETTINGS = {
    'max_epochs': 1,
    'max_dataset_size': 500,
}

BENCHMARKS = [
    # name, is_seq_output, output_type
    ('signalP_binary', False, 'binary'),
    ('fluorescence', False, 'numeric'),
    ('remote_homology', False, 'categorical'),
    ('stability', False, 'numeric'),
    ('scop', False, 'categorical'),
    ('secondary_structure', True, 'categorical'),
    ('disorder_secondary_structure', True, 'categorical'),
    ('ProFET_NP_SP_Cleaved', False, 'binary'),
    ('phosphoserine', True, 'categorical'),
    ('PhosphositePTM', True, 'categorical'),
]

class ModelSpecification:
    
    def __init__(self, create_model_function, model_weights = None, pretrained_model_weigths_file_path = None, create_model_kwargs = {}):
        assert model_weights is None or pretrained_model_weigths_file_path is None, 'Please choose up to one way of specifying model weights.'
        self.create_model_function = create_model_function
        self.model_weights = model_weights
        self.pretrained_model_weigths_file_path = pretrained_model_weigths_file_path
        self.create_model_kwargs = create_model_kwargs
        
    def create_model(self, seq_len):
        
        import tensorflow.keras.backend as K
        K.clear_session()
        
        model = self.create_model_function(seq_len, n_tokens, N_ANNOTATIONS, **self.create_model_kwargs)
        
        if self.model_weights is not None:
            model.set_weights(self.model_weights)
        elif self.pretrained_model_weigths_file_path is not None:
            with open(self.pretrained_model_weigths_file_path, 'rb') as f:
                model_weights, optimizer_weights = pickle.load(f)
                model.set_weights(model_weights)
        
        return model        

class OutputType:
    
    def __init__(self, is_seq, output_type):
        self.is_seq = is_seq
        self.output_type = output_type
        self.is_numeric = (output_type == 'numeric')
        self.is_binary = (output_type == 'binary')
        self.is_categorical = (output_type == 'categorical')
        
    def __str__(self):
        if self.is_seq:
            return '%s sequence' % self.output_type
        else:
            return 'global %s' % self.output_type
            
def resolve_settings(settings = {}, debug = False):
    
    final_settings = dict(BASELINE_SETTINGS)
    
    if debug:
        final_settings.update(DEBUG_SETTINGS)
        
    final_settings.update(settings)
    return final_settings
        
def get_benchmark_output_type(benchmark_name):
    for name, is_seq_output, output_type in BENCHMARKS:
        if name == benchmark_name:
            return OutputType(is_seq_output, output_type)
            
def tokenize_seqs(seqs, seq_len):
    '''
    Note that tokenize_seq already adds <START> and <END> tokens.
    '''
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)
    
def encode_X(raw_seq, seq_len):
    return [
        tokenize_seqs(raw_seq, seq_len),
        np.zeros((len(raw_seq), N_ANNOTATIONS), dtype = np.int8)
    ]
    
def encode_seq_Y(raw_Y, unique_labels, seq_len):

    '''
    Note that <START> and <END> correspond to padding in Y.
    '''
    
    n_labels = len(unique_labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    Y = np.zeros((len(raw_Y), seq_len), dtype = int)
    
    for i, y in enumerate(raw_Y):
        for j, label in enumerate(y):
            # +1 to account for the <START> token at the beginning.
            Y[i, j + 1] = label_to_index[label]
        
        Y[i, [0] + list(range(len(y) + 1, seq_len))] = n_labels
    
    return Y
    
def encode_categorical_Y(raw_Y, unique_labels):
    
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    Y = np.zeros(len(raw_Y), dtype = int)
    
    for i, y in enumerate(raw_Y):
        Y[i] = label_to_index[y]
        
    return Y
    
def encode_Y(raw_Y, output_type, unique_labels, seq_len):
    if output_type.is_seq:
        return encode_seq_Y(raw_Y, unique_labels, seq_len)
    elif output_type.is_categorical:
        return encode_categorical_Y(raw_Y, unique_labels)
    elif output_type.is_numeric or output_type.is_binary:
        return raw_Y.values.astype(float)
    else:
        raise ValueError('Unexpected output type: %s' % output_type)
            
def load_benchmark_dataset(benchmark_name):
    
    text_file_path = os.path.join(BENCHMARKS_DIR, '%s.benchmark.txt' % benchmark_name)    
    train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.train.csv' % benchmark_name)
    valid_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.valid.csv' % benchmark_name)
    test_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.test.csv' % benchmark_name)
    
    train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
    test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()
          
    if os.path.exists(valid_set_file_path):
        valid_set = pd.read_csv(valid_set_file_path).dropna().drop_duplicates()
    else:
        
        log(f'Validation set {valid_set_file_path} missing. Splitting training set instead.')
        
        try: 
            train_set, valid_set = train_test_split(train_set, stratify = train_set['labels'], test_size = 0.1, random_state = 0)
        except:
            log('Stratification did not work, randomly sampling instead.')
            train_set, valid_set = train_test_split(train_set, test_size = 0.1, random_state = 0)    
    
    return text_file_path, train_set, valid_set, test_set
    
def filter_dataset_by_len(dataset, name, settings):
    max_allowed_input_seq_len = settings['seq_len'] - 2
    filtered_dataset = dataset[dataset['text'].str.len() <= max_allowed_input_seq_len]
    n_removed_records = len(dataset) - len(filtered_dataset)
    log('%s: Filtered out %d of %d (%.1f%%) records of lengths exceeding %d.' % (name, n_removed_records, len(dataset), 100 * n_removed_records / len(dataset), \
            max_allowed_input_seq_len))
    return filtered_dataset

def preprocess_benchmark_dataset(train_set, valid_set, test_set, output_type, unique_labels, seq_len):

    train_X = encode_X(train_set['text'], seq_len)
    valid_X = encode_X(valid_set['text'], seq_len)
    test_X = encode_X(test_set['text'], seq_len)
    
    train_Y = encode_Y(train_set['labels'], output_type, unique_labels, seq_len)
    valid_Y = encode_Y(valid_set['labels'], output_type, unique_labels, seq_len)
    test_Y = encode_Y(test_set['labels'], output_type, unique_labels, seq_len)
        
    return train_X, valid_X, test_X, train_Y, valid_Y, test_Y

def evaluate(y_pred, y_true, output_type, unique_labels):

    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, log_loss, accuracy_score, f1_score, precision_score, recall_score, \
            matthews_corrcoef, classification_report, confusion_matrix
            
    if output_type.is_numeric:
        assert not output_type.is_seq
        log('R2: %.3f' % r2_score(y_true, y_pred))
        log('Mean absolute error: %.3f' % mean_absolute_error(y_true, y_pred))
        log('Spearman\'s rank correlation: %.3f' % spearmanr(y_true, y_pred)[0])
    else:
    
        if output_type.is_seq:
            unique_labels = unique_labels + ['<PAD>']
            
        n_unique_labels = len(unique_labels)
        str_unique_labels = list(map(str, unique_labels))
    
        y_pred_classes = y_pred.argmax(axis = -1)
        
        spread_y_true = y_true.flatten()
        spread_y_pred_classes = y_pred_classes.flatten()
        spread_y_pred = y_pred.reshape((y_pred.size // y_pred.shape[-1], y_pred.shape[-1]))
        
        if output_type.is_seq:
            spread_no_padding_mask = (spread_y_true != n_unique_labels - 1)
        else:
            spread_no_padding_mask = np.ones_like(spread_y_true, dtype = bool)
            
        spread_y_true_no_padding = spread_y_true[spread_no_padding_mask]
        spread_y_pred_classes_no_padding = spread_y_pred_classes[spread_no_padding_mask]
        spread_y_pred_no_padding = spread_y_pred[spread_no_padding_mask]
        
        if output_type.is_binary:
            log('AUC: %.3f' % roc_auc_score(spread_y_true, spread_y_pred))
            log('MCC: %.3f' % matthews_corrcoef(spread_y_true, spread_y_pred_classes))
                
        log('Log loss: %.2f%%' % (100 * log_loss(spread_y_true_no_padding, spread_y_pred_no_padding, labels = unique_labels)))
        log('Accuracy: %.2f%%' % (100 * accuracy_score(spread_y_true_no_padding, spread_y_pred_classes_no_padding)))
        log('F1 (micro avg.): %.2f%%' % (100 * f1_score(spread_y_true_no_padding, spread_y_pred_classes_no_padding, average = 'micro')))
        log('Precision (micro avg.): %.2f%%' % (100 * precision_score(spread_y_true_no_padding, spread_y_pred_classes_no_padding, average = 'micro')))
        log('Recall (micro avg.): %.2f%%' % (100 * recall_score(spread_y_true_no_padding, spread_y_pred_classes_no_padding, average = 'micro')))   

        with pd.option_context('display.max_rows', 16, 'display.max_columns', 10):
            log('Classification report:')
            display(pd.DataFrame(classification_report(spread_y_true, spread_y_pred_classes, labels = np.arange(n_unique_labels), \
                    target_names = str_unique_labels, output_dict = True)).transpose())
            log('Confusion matrix:')
            display(pd.DataFrame(confusion_matrix(spread_y_true, spread_y_pred_classes, labels = np.arange(n_unique_labels)), index = str_unique_labels, \
                    columns = str_unique_labels))
        
def build_fine_tuning_model(model_specification, settings, output_type, unique_labels):

    model = model_specification.create_model(settings['seq_len'])
    input_seq_layer, input_annoatations_layer = model.input
    output_seq_layer, output_annoatations_layer = model.output

    if output_type.is_seq:
        # A sequence is always treated as a categorical output, with an additional padding label.
        output_layer = keras.layers.Dense(len(unique_labels) + 1, activation = 'softmax')(output_seq_layer)
        loss = 'sparse_categorical_crossentropy'
    else:
        if output_type.is_categorical:
            output_layer = keras.layers.Dense(len(unique_labels), activation = 'softmax')(output_annoatations_layer)
            loss = 'sparse_categorical_crossentropy'
        elif output_type.is_binary:
            output_layer = keras.layers.Dense(1, activation = 'sigmoid')(output_annoatations_layer)
            loss = 'binary_crossentropy'
        elif output_type.is_numeric:
            output_layer = keras.layers.Dense(1, activation = None)(output_annoatations_layer)
            loss = 'mse'
        else:
            raise ValueError('Unexpected output type: %s' % output_type)
            
    model = keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = output_layer)
    model.compile(loss = loss, optimizer = keras.optimizers.Adam(lr = settings['initial_lr'])) 
    return model

def train_and_eval(model_specification, settings, train_set, valid_set, test_set, output_type, unique_labels):

    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = preprocess_benchmark_dataset(train_set, valid_set, test_set, output_type, unique_labels, settings['seq_len'])
    model = build_fine_tuning_model(model_specification, settings, output_type, unique_labels)
    
    model.fit(train_X, train_Y, batch_size = settings['batch_size'], epochs = settings['max_epochs'], validation_data = (valid_X, valid_Y), \
            callbacks = settings['training_callbacks'])
    
    log('*** Training-set performance: ***')
    train_Y_pred = model.predict(train_X, batch_size = settings['batch_size'])
    evaluate(train_Y_pred, train_Y, output_type, unique_labels)
    
    log('*** Validation-set performance: ***')
    valid_Y_pred = model.predict(valid_X, batch_size = settings['batch_size'])
    evaluate(valid_Y_pred, valid_Y, output_type, unique_labels)
    
    log('*** Test-set performance: ***')
    test_Y_pred = model.predict(test_X, batch_size = settings['batch_size'])
    evaluate(test_Y_pred, test_Y, output_type, unique_labels)
                                                         
def run_benchmark(benchmark_name, model_specification, settings = {}, debug = False):
    
    log('========== %s ==========' % benchmark_name)  

    settings = resolve_settings(settings, debug)
    
    output_type = get_benchmark_output_type(benchmark_name)
    log('Output type: %s' % output_type)
    
    _, train_set, valid_set, test_set = load_benchmark_dataset(benchmark_name)        
    log(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')
    
    train_set = filter_dataset_by_len(train_set, 'Training set', settings)
    valid_set = filter_dataset_by_len(valid_set, 'Validation set', settings)
    test_set = filter_dataset_by_len(test_set, 'Test set', settings)
    
    if settings['max_dataset_size'] is not None:
        log('Limiting the training, validation and test sets to %d records each.' % settings['max_dataset_size'])
        train_set = train_set.sample(min(settings['max_dataset_size'], len(train_set)), random_state = 0)
        valid_set = valid_set.sample(min(settings['max_dataset_size'], len(valid_set)), random_state = 0)
        test_set = test_set.sample(min(settings['max_dataset_size'], len(test_set)), random_state = 0)
    
    if output_type.is_categorical:
        train_set['labels'] = train_set['labels'].astype(str)
        valid_set['labels'] = valid_set['labels'].astype(str)
        test_set['labels'] = test_set['labels'].astype(str)
    else:
        train_set['labels'] = train_set['labels'].astype(float)
        valid_set['labels'] = valid_set['labels'].astype(float)
        test_set['labels'] = test_set['labels'].astype(float)
      
    if output_type.is_categorical:
        
        if output_type.is_seq:
            unique_labels = sorted(set.union(*train_set['labels'].apply(set)) | set.union(*valid_set['labels'].apply(set)) | set.union(*test_set['labels'].apply(set)))
        else:
            unique_labels = sorted(set(train_set['labels'].unique()) | set(valid_set['labels'].unique()) | set(test_set['labels'].unique()))
            
        log('%d unique lebels.' % len(unique_labels))
    elif output_type.is_binary:
        unique_labels = [0, 1]
    else:
        unique_labels = None
        
    train_and_eval(model_specification, settings, train_set, valid_set, test_set, output_type, unique_labels)

def run_all_benchmarks(model_specification, settings = {}, debug = False):
    
    for benchmark_name, _, _ in BENCHMARKS:
        run_benchmark(benchmark_name, model_specification, settings = settings, debug = debug)
        
    log('Done.')
