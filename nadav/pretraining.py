import os
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import h5py

from tensorflow import keras

from pwas.shared_utils.util import log

from tokenization import ADDED_TOKENS_PER_SEQ, additional_token_to_index, n_tokens, tokenize_seq
from model_util import get_model_creator, recreate_model_with_same_state, save_model_state, load_model_state

H5_FILE_PATH = '/cs/phd/nadavb/cafa_project/data/dataset.h5'

DEFAULT_EPISODE_SETTINGS = [
    # max_seq_len, batch_size
    (100, 128),
    (450, 64),
    (1200, 32),
]

class SampleCache:
    
    def __init__(self, seqs = [], annotation_masks = [], test_set_flags = []):
        self.seqs = list(seqs)
        self.annotation_masks = list(annotation_masks)
        self.test_set_flags = list(test_set_flags)
        
    def extend(self, other_cache):
        self.seqs.extend(other_cache.seqs)
        self.annotation_masks.extend(other_cache.annotation_masks)
        self.test_set_flags.extend(other_cache.test_set_flags)
        
    def pop(self, n):
        popped_sample_cache = self.slice_first(n)
        self.seqs = self.seqs[n:]
        self.annotation_masks = self.annotation_masks[n:]
        self.test_set_flags = self.test_set_flags[n:]
        return popped_sample_cache
    
    def slice_first(self, n):
        return SampleCache(self.seqs[:n], self.annotation_masks[:n], self.test_set_flags[:n])
        
    def slice_indices(self, indices):
        return SampleCache([self.seqs[i] for i in indices], [self.annotation_masks[i] for i in indices], \
                [self.test_set_flags[i] for i in indices])
    
    def __len__(self):
        assert len(self.seqs) == len(self.annotation_masks) == len(self.test_set_flags)
        return len(self.seqs)
    
class DatasetHandler:
    
    def __init__(self, dataset_h5f):
        self.dataset_h5f = dataset_h5f
        self.total_size = len(dataset_h5f['seq_lengths'])
        
    def __getitem__(self, slicing):
        return SampleCache(self.dataset_h5f['seqs'][slicing], self.dataset_h5f['annotation_masks'][slicing], \
                self.dataset_h5f['test_set_similarity'][slicing])

class EpisodeDataManager:
    
    def __init__(self, max_seq_len, batch_size, batches_per_epoch):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.epoch_size = self.batches_per_epoch * self.batch_size
        self.sample_cache = SampleCache()
        
    def is_epoch_ready(self, n_required_samples = None):
        return len(self.sample_cache) >= self._resolve_epoch_size(n_required_samples)
    
    def get_next_raw_epoch(self, size = None):
        return self.sample_cache.pop(self._resolve_epoch_size(size))
    
    def peek_raw_epoch(self, size = None):
        return self.sample_cache.slice_first(self._resolve_epoch_size(size))
    
    def encode_next_epoch(self, log_length_dist = True):
        
        seq_lengths, encoded_seqs, encoded_annotation_masks = self._encode_epoch(self.get_next_raw_epoch())
        
        if log_length_dist:
            log('Epoch sequence length distribution (for max_seq_len = %d): %s' % (self.max_seq_len, \
                    ', '.join('%s: %s' % item for item in pd.Series(seq_lengths).describe().iteritems())))
        
        return encoded_seqs, encoded_annotation_masks
    
    def encode_dummy_epoch(self, size = 1):
        seq_lengths, encoded_seqs, encoded_annotation_masks = self._encode_epoch(self.peek_raw_epoch(size))
        return encoded_seqs, encoded_annotation_masks
    
    def _encode_epoch(self, epoch_sample_cache):
        
        pad_token_index = additional_token_to_index['<PAD>']
        tokenized_seqs = list(map(tokenize_seq, epoch_sample_cache.seqs))
        seq_lengths = np.array(list(map(len, tokenized_seqs)))
        max_offsets = np.maximum(seq_lengths - self.max_seq_len, 0)
        chosen_offsets = (np.random.rand(self.epoch_size) * (max_offsets + 1)).astype(int)
        trimmed_tokenized_seqs = [seq_tokens[chosen_offset:(chosen_offset + self.max_seq_len)] for seq_tokens, chosen_offset in \
                zip(tokenized_seqs, chosen_offsets)]
        encoded_seqs = np.array([seq_tokens + max(self.max_seq_len - len(seq_tokens), 0) * [pad_token_index] for seq_tokens in \
                trimmed_tokenized_seqs]).astype(np.int8)
        
        encoded_annotation_masks = np.concatenate([annotation_mask.reshape(1, -1) for annotation_mask in \
                epoch_sample_cache.annotation_masks], axis = 0).astype(bool)
        encoded_annotation_masks[epoch_sample_cache.test_set_flags, :] = False
        
        return seq_lengths, encoded_seqs, encoded_annotation_masks
    
    def _resolve_epoch_size(self, size):
        if size is None:
            return self.epoch_size
        else:
            return size
    
class EpochGenerator:
    
    def __init__(self, batches_per_epoch = 100, p_seq_noise = 0.05, p_no_input_annot = 0.5, p_annot_noise_positive = 0.25, \
            p_annot_noise_negative = 1e-04, load_chunk_size = 100000, min_time_per_episode = timedelta(minutes = 15), \
            episode_settings = DEFAULT_EPISODE_SETTINGS):
        
        self.batches_per_epoch = batches_per_epoch
        self.p_seq_noise = p_seq_noise
        self.p_no_input_annot = p_no_input_annot
        self.p_annot_noise_positive = p_annot_noise_positive
        self.p_annot_noise_negative = p_annot_noise_negative
        self.load_chunk_size = load_chunk_size
        self.min_time_per_episode = min_time_per_episode
        
        self.episode_managers = [EpisodeDataManager(max_seq_len, batch_size, self.batches_per_epoch) for max_seq_len, batch_size in \
                episode_settings]
        self.episode_max_seq_lens = np.array([episode_manager.max_seq_len for episode_manager in self.episode_managers])
        
    def setup(self, dataset_handler, starting_sample_index = 0):
        self.dataset_handler = dataset_handler
        self.current_sample_index = starting_sample_index % self.dataset_handler.total_size
        self._load_chunk()
        self._select_new_episode()
        return self._current_episode
    
    def determine_episode_and_ready_next_epoch(self):
        
        if self._episode_selection_time + self.min_time_per_episode <= datetime.now():
            old_episode = self._current_episode
            self._select_new_episode()
            changed_episode = (self._current_episode is not old_episode)
        else:
            changed_episode = False
            
        while not self._current_episode.is_epoch_ready():
            self._load_chunk()

        return changed_episode, self._current_episode
        
    def create_next_epoch(self):
        return self._encode_epoch(*self.create_next_epoch_X())
        
    def create_dummpy_epoch(self, size = 1):
        return self._encode_epoch(*self.create_next_dummy_epoch_X(size))
        
    def create_next_epoch_X(self):
        assert self._current_episode.is_epoch_ready()
        return self._current_episode.encode_next_epoch()
    
    def create_next_dummy_epoch_X(self, size = 1):
        
        while not self._current_episode.is_epoch_ready(size):
            self._load_chunk()
            
        return self._current_episode.encode_dummy_epoch(size)
    
    def _select_new_episode(self):
        self._current_episode = max(self.episode_managers, key = lambda episode_manager: len(episode_manager.sample_cache))
        self._episode_selection_time = datetime.now()
            
    def _load_chunk(self):
        
        chunk_sample_cache = self.dataset_handler[self.current_sample_index:(self.current_sample_index + self.load_chunk_size)]
        self.current_sample_index += self.load_chunk_size
        
        if self.current_sample_index >= self.dataset_handler.total_size:
            self.current_sample_index = 0
            
        self._assign_samples(chunk_sample_cache)
        
    def _assign_samples(self, sample_cache):
        
        seq_lens = np.array(list(map(len, sample_cache.seqs))) + ADDED_TOKENS_PER_SEQ
        assigned_episode_indices = self._select_episodes_to_assign(seq_lens)
        
        for episode_manager_index, episode_manager in enumerate(self.episode_managers):
            sample_indices_for_episode, = np.where(assigned_episode_indices == episode_manager_index)
            episode_manager.sample_cache.extend(sample_cache.slice_indices(sample_indices_for_episode))
        
    def _select_episodes_to_assign(self, seq_lens, gamma = 1):
        # The smaller the distance between a sample's sequence length to an episode's maximum sequence length, the higher the chance
        # that it will be assigned to that episode.
        samples_by_episodes_seq_len_ratio = seq_lens.reshape(-1, 1) / self.episode_max_seq_lens.reshape(1, -1)
        samples_by_episodes_seq_len_symmetric_ratio = np.maximum(samples_by_episodes_seq_len_ratio, 1 / samples_by_episodes_seq_len_ratio)
        raw_samples_by_episodes_probs = np.exp(-gamma * samples_by_episodes_seq_len_symmetric_ratio)
        samples_by_episodes_probs = raw_samples_by_episodes_probs / raw_samples_by_episodes_probs.sum(axis = -1).reshape(-1, 1)
        samples_by_episodes_cum_probs = samples_by_episodes_probs.cumsum(axis = -1)
        assigned_episode_indices = (np.random.rand(len(seq_lens), 1) <= samples_by_episodes_cum_probs).argmax(axis = 1)
        return assigned_episode_indices
    
    def _encode_epoch(self, encoded_seqs, encoded_annotation_masks):
        
        seqs_noise_mask = np.random.choice([True, False], encoded_seqs.shape, p = [1 - self.p_seq_noise, self.p_seq_noise])
        random_seq_tokens = np.random.randint(0, n_tokens, encoded_seqs.shape)
        noisy_encoded_seqs = np.where(seqs_noise_mask, encoded_seqs, random_seq_tokens)

        noisy_annotations_when_positive = np.random.choice([True, False], encoded_annotation_masks.shape, \
                p = [1 - self.p_annot_noise_positive, self.p_annot_noise_positive])
        noisy_annotations_when_negative = np.random.choice([True, False], encoded_annotation_masks.shape, \
                p = [self.p_annot_noise_negative, 1 - self.p_annot_noise_negative])
        noisy_annotation_masks = np.where(encoded_annotation_masks, noisy_annotations_when_positive, \
                noisy_annotations_when_negative)
        noisy_annotation_masks[np.random.choice([True, False], len(noisy_annotation_masks), p = [self.p_no_input_annot, \
                1 - self.p_no_input_annot]), :] = False

        # When a protein has no annotations at all, we don't know whether it's because such annotations don't exist or just not found,
        # so it's safer to set the loss weight of those annotations to zero.
        seq_weights = np.ones(len(encoded_seqs))
        annotation_weights = encoded_annotation_masks.any(axis = -1).astype(float)
        
        X = [noisy_encoded_seqs, noisy_annotation_masks.astype(np.int8)]
        Y = [np.expand_dims(encoded_seqs, axis = -1), encoded_annotation_masks.astype(np.int8)]
        sample_weigths = [seq_weights, annotation_weights]
        
        return X, Y, sample_weigths
    
class AutoSaveManager:
    
    def __init__(self, directory, every_epochs_to_save = 10, every_saves_to_keep = 25):
        self.directory = directory
        self.every_epochs_to_save = every_epochs_to_save
        self.every_saves_to_keep = every_saves_to_keep
        self.last_saved_path_to_delete = None
        self.n_saves = 0
    
    def on_epoch_end(self, model, epoch_index, sample_index):
        
        if epoch_index % self.every_epochs_to_save != 0:
            return
        
        save_path = os.path.join(self.directory, 'epoch_%d_sample_%d.pkl' % (epoch_index, sample_index))
        save_model_state(model, save_path)
        self.n_saves += 1
        
        if self.last_saved_path_to_delete is not None:
            os.remove(self.last_saved_path_to_delete)
            
        if self.n_saves % self.every_saves_to_keep == 0:
            self.last_saved_path_to_delete = None
        else:
            self.last_saved_path_to_delete = save_path
    
class ModelTrainer:
    
    def __init__(self, model_creator, epoch_generator, autosave_manager = None, weights_dir = None, lr = 2e-04, annots_loss_weight = 1e03, fit_callbacks = []):
        self.model_creator = model_creator
        self.epoch_generator = epoch_generator
        self.autosave_manager = autosave_manager
        self.weights_dir = weights_dir
        self.lr = lr
        self.annots_loss_weight = annots_loss_weight
        self.fit_callbacks = fit_callbacks
        
    def setup(self, dataset_handler, resume_epoch = None):
        
        if resume_epoch is None:
            self.current_epoch_index = 0
            starting_sample_index = 0
            resumed_weights_file_path = None
        else:
            self.current_epoch_index, starting_sample_index = resume_epoch
            self.current_epoch_index += 1
            resumed_weights_file_path = os.path.join(self.weights_dir, 'epoch_%d_sample_%d.pkl' % resume_epoch)
        
        starting_episode = self.epoch_generator.setup(dataset_handler, starting_sample_index)
        log('Starting with episode with max_seq_len = %d.' % starting_episode.max_seq_len)
        
        self.model = self._get_create_and_compile_model_function(starting_episode.max_seq_len, will_weights_be_reloaded = \
                (resumed_weights_file_path is not None))()
        self.model.summary()
        
        if resumed_weights_file_path is not None:
            load_model_state(self.model, resumed_weights_file_path)
            log('Loaded weights from %s.' % resumed_weights_file_path)
        
    def train_next_epoch(self, autosave_if_needed = True):
        
        changed_episode, episode = self.epoch_generator.determine_episode_and_ready_next_epoch()
        
        if changed_episode:
            log('Starting a new episode with max_seq_len = %d.' % episode.max_seq_len)
            self.model = recreate_model_with_same_state(self.model, self._get_create_and_compile_model_function(episode.max_seq_len, \
                    will_weights_be_reloaded = True))
        
        X, Y, sample_weigths = self.epoch_generator.create_next_epoch()
        log('Epoch %d (current sample %d):' % (self.current_epoch_index, self.epoch_generator.current_sample_index))
        self.model.fit(X, Y, sample_weight = sample_weigths, batch_size = episode.batch_size, callbacks = self.fit_callbacks)
        
        if autosave_if_needed and self.autosave_manager is not None:
            self.autosave_manager.on_epoch_end(self.model, self.current_epoch_index, self.epoch_generator.current_sample_index)
            
        self.current_epoch_index += 1
        
    def train(self, n_epochs = None, autosave = True):
        for _ in (itertools.count() if n_epochs is None else range(n_epochs)):
            self.train_next_epoch(autosave_if_needed = autosave)
            
    def _get_create_and_compile_model_function(self, max_seq_len, will_weights_be_reloaded = False):
        
        def create_and_compile_model_function():
            
            model = self.model_creator(max_seq_len)
            self._compile_model(model)
            
            if will_weights_be_reloaded:
                self._train_for_a_dummy_epoch(model)
            
            return model
        
        return create_and_compile_model_function
    
    def _compile_model(self, model):
        model.compile(optimizer = keras.optimizers.Adam(lr = self.lr), loss = ['sparse_categorical_crossentropy', \
                'binary_crossentropy'], loss_weights = [1, self.annots_loss_weight])
        
    def _train_for_a_dummy_epoch(self, model):
        '''
        For some reason keras requires this strange little hack in order to properly initialize a new model's optimizer, so that
        the optimizer's weights can be reloaded from an existing state.
        '''
        X, Y, sample_weigths = self.epoch_generator.create_dummpy_epoch(size = 1)
        model.fit(X, Y, batch_size = 1, verbose = 0)

def run_pretraining(create_model_function, epoch_generator, autosave_manager = None, weights_dir = None, resume_epoch = None, n_epochs = None, \
        lr = 2e-04, annots_loss_weight = 1e03, fit_callbacks = [], model_creation_kwargs = {}):

    np.random.seed(0)
    
    with h5py.File(H5_FILE_PATH, 'r') as h5f:
        included_annotation_indices = h5f['included_annotation_indices'][:]
    
    n_annotations = len(included_annotation_indices)
    print('%d unique annotations.' % n_annotations)
    
    model_creator = get_model_creator(n_annotations, create_model_function, **model_creation_kwargs)
    model_trainer = ModelTrainer(model_creator, epoch_generator, autosave_manager = autosave_manager, weights_dir = weights_dir, lr = lr, \
            annots_loss_weight = annots_loss_weight, fit_callbacks = fit_callbacks)

    with h5py.File(H5_FILE_PATH, 'r') as h5f:
        model_trainer.setup(DatasetHandler(h5f), resume_epoch = resume_epoch)
        model_trainer.train(n_epochs = n_epochs)
        
    return model_trainer

