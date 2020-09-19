

import os
import h5py
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display
from pwas.shared_utils.util import log
from datetime import datetime, timedelta

from training_utils import *
from config import *

if PRETRAINING_FIXED_SEED: np.random.seed(0)

mkdir_if_not_exists(os.path.join(PRETRAINING_BASE_WEIGHTS_DIR, 'weights'))
mkdir_if_not_exists(os.path.join(os.path.join(PRETRAINING_BASE_WEIGHTS_DIR, 'weights'), 'autosave'))
with h5py.File(H5_FILE_PATH, 'r') as h5f: included_annotation_indices = h5f['included_annotation_indices'][:]
n_annotations = len(included_annotation_indices)
print('%d unique annotations.' % n_annotations)

def recreate_model_with_same_state(old_model, create_and_compile_model_funtion):
    model_weights, optimizer_weights = old_model.get_weights(), old_model.optimizer.get_weights()
    new_model = create_and_compile_model_funtion()
    new_model.set_weights(model_weights)
    new_model.optimizer.set_weights(optimizer_weights)
    return new_model

class ModelTrainer:

    def __init__(self, epoch_generator, autosave_manager, lr=1e-04, annots_loss_weight=1e04):
        self.epoch_generator = epoch_generator
        self.autosave_manager = autosave_manager
        self.lr = lr
        self.annots_loss_weight = annots_loss_weight

    def setup(self, dataset_handler, resume_epoch=None):
        if resume_epoch is None:
            self.current_epoch_index = 0
            starting_sample_index = 0
            resumed_weights_file_path = None
        else:
            self.current_epoch_index, starting_sample_index = resume_epoch
            self.current_epoch_index += 1
            resumed_weights_file_path = os.path.join(os.path.join(PRETRAINING_BASE_WEIGHTS_DIR, 'weights'), 'epoch_%d_sample_%d.pkl' % resume_epoch)
        starting_episode = self.epoch_generator.setup(dataset_handler, starting_sample_index)
        log('Starting with episode with max_seq_len = %d.' % starting_episode.max_seq_len)
        self.model = self._get_create_and_compile_model_function(starting_episode.max_seq_len, will_weights_be_reloaded=(resumed_weights_file_path is not None))()
        self.model.summary()
        if resumed_weights_file_path is not None:
            load_model(self.model, resumed_weights_file_path, load_optimizer_weights=True)
            log('Loaded weights from %s.' % resumed_weights_file_path)

    def train_next_epoch(self, autosave_if_needed=True):
        changed_episode, episode = self.epoch_generator.determine_episode_and_ready_next_epoch()
        if changed_episode:
            log('Starting a new episode with max_seq_len = %d.' % episode.max_seq_len)
            self.model = recreate_model_with_same_state(self.model, self._get_create_and_compile_model_function(episode.max_seq_len, will_weights_be_reloaded=True))
        X, Y, sample_weigths = self.epoch_generator.create_next_epoch()
        log('Epoch %d (current sample %d):' % (self.current_epoch_index, self.epoch_generator.current_sample_index))
        self.model.fit(X, Y, sample_weight=sample_weigths, batch_size=episode.batch_size)
        if autosave_if_needed: self.autosave_manager.on_epoch_end(self.model, self.current_epoch_index, self.epoch_generator.current_sample_index)
        self.current_epoch_index += 1

    def train_forever(self, autosave=True):
        while True: self.train_next_epoch(autosave_if_needed=autosave)

    def _get_create_and_compile_model_function(self, max_seq_len, will_weights_be_reloaded=False):
        def create_and_compile_model_function():
            model = MODEL_TYPE.create_model(max_seq_len, n_tokens, N_ANNOTATIONS)
            self._compile_model(model)
            if will_weights_be_reloaded:
                self._train_for_a_dummy_epoch(model)
            return model
        return create_and_compile_model_function

    def _compile_model(self, model):
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
                      loss_weights=[1, self.annots_loss_weight])

    def _train_for_a_dummy_epoch(self, model):
        X, Y, sample_weigths = self.epoch_generator.create_dummpy_epoch(size=1)
        model.fit(X, Y, batch_size=1, verbose=0)

epoch_generator = EpochGenerator(n_tokens)
autosave_manager = AutoSaveManager(os.path.join(os.path.join(PRETRAINING_BASE_WEIGHTS_DIR, 'weights'), 'autosave'), every_epochs_to_save = PRETRAINING_SAVE_EVERY)
model_trainer = ModelTrainer(epoch_generator, autosave_manager)

with h5py.File(H5_FILE_PATH, 'r') as h5f:
    model_trainer.setup(DatasetHandler(h5f), resume_epoch = PRETRAINING_RESUME_EPOCH)
    model_trainer.train_forever()

with h5py.File(H5_FILE_PATH, 'r') as h5f:
    test_set_indices, = np.where(h5f['test_set_similarity'][:])
    chosen_sample_indices = list(sorted(np.random.choice(test_set_indices, PRETRAINING_N)))
    del test_set_indices
    chosen_samples = DatasetHandler(h5f)[chosen_sample_indices]
    chosen_uniprot_ids = h5f['uniprot_ids'][chosen_sample_indices]
    
chosen_samples.test_set_flags = [False for _ in range(PRETRAINING_N)]
seq_lens = np.array(list(map(len, chosen_samples.seqs)))
model_max_seq_len = MAX_GLOBAL_SEQ_LEN
episode = EpisodeDataManager(model_max_seq_len, batch_size = 1, batches_per_epoch = 1)
episode.sample_cache = chosen_samples
epoch_generator._current_episode = episode
model = recreate_model_with_same_state(model_trainer.model, model_trainer._get_create_and_compile_model_function(model_max_seq_len, will_weights_be_reloaded = True))
    
for uniprot_id in chosen_uniprot_ids:
    
    print('UniProt ID: %s (https://www.uniprot.org/uniprot/%s)' % (uniprot_id, uniprot_id))

    X, Y_true, _ = epoch_generator._encode_epoch(*episode.encode_next_epoch())
    Y_pred = model.predict(X)
    
    X_seqs, X_annots = X
    Y_true_seqs, Y_true_annots = Y_true
    Y_pred_seqs, Y_pred_annots = Y_pred

    X_seqs = X_seqs.flatten()
    X_annots = X_annots.flatten()
    Y_true_seqs = Y_true_seqs.flatten()
    Y_true_annots = Y_true_annots.flatten()
    Y_pred_seqs = Y_pred_seqs[0, :, :]
    Y_pred_annots = Y_pred_annots.flatten()
    Y_pred_seqs_max = Y_pred_seqs.argmax(axis = -1)

    seq_result = pd.DataFrame()
    seq_result['true'] = list(map(index_to_token.get, Y_true_seqs))
    seq_result['input'] = list(map(index_to_token.get, X_seqs))
    seq_result['max'] = list(map(index_to_token.get, Y_pred_seqs_max))
    seq_result['p_true'] = Y_pred_seqs[np.arange(model_max_seq_len), Y_true_seqs]
    seq_result['p_input'] = Y_pred_seqs[np.arange(model_max_seq_len), X_seqs]
    seq_result['p_max'] = Y_pred_seqs[np.arange(model_max_seq_len), Y_pred_seqs_max]

    print('Sequence results:')

    with pd.option_context('display.max_columns', model_max_seq_len):
        display(seq_result[(seq_result['true'] != seq_result['input']) | (seq_result['p_true'] < 0.9)].transpose())

    true_annots, = np.where(Y_true_annots)
    input_annots, = np.where(X_annots)
    relevant_annots = sorted(set(true_annots) | set(input_annots) | set(np.where(Y_pred_annots >= 0.05)[0]))

    print('Annotation results:')
    print('True annotations: %s' % true_annots)
    print('Input annotations: %s' % input_annots)
    print('Predicted annotations: %s' % ', '.join('%d (%.2g)' % (annot, Y_pred_annots[annot]) for annot in relevant_annots))

def create_and_compile_model_function():
    global model_max_seq_len, model_trainer, X, Y_true
    model = MODEL_TYPE.create_model(model_max_seq_len, n_tokens, N_ANNOTATIONS)
    model_trainer._compile_model(model)
    model.fit(X, Y_true, batch_size = 1, verbose = 0)
    return model

def train_model(model):
    global X, Y_true
    for _ in range(10):
        model.fit(X, Y_true, batch_size = 1, verbose = 0)

train_model(model)
model1 = recreate_model_with_same_state(model, create_and_compile_model_function)
train_model(model1)
Y_pred1 = model1.predict(X)
model2 = recreate_model_with_same_state(model, create_and_compile_model_function)
train_model(model2)
Y_pred2 = model2.predict(X)
for y_pred1, y_pred2 in zip(Y_pred1, Y_pred2): print(np.abs(y_pred1 - y_pred2).max())
