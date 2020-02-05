
import os
import re
import sys
import json
import keras
import random
import codecs
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from tqdm import trange, tqdm
from keras.models import Model
from keras import backend as K
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert.keras_bert.bert import get_model
from keras_bert import gen_batch_inputs, get_base_dict, compile_model
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda
from keras_bert.loader import load_trained_model_from_checkpoint, build_model_from_config

def pad_tokens(tokens, max_len, PAD_TOKEN): return np.concatenate([tokens, np.full(max_len - len(tokens), PAD_TOKEN)])
def encode_labels(labels, label_to_index, n_labels):
    encoded_labels = np.zeros(n_labels, dtype=np.int32)
    for label in labels: encoded_labels[label_to_index[label]] += 1
    return encoded_labels

def get_bert_generator(token_dict, annotations, token_list, MAX_SEQ_TOKEN_LEN, PAD_TOKEN, seq_len=20, mask_rate=0.3, swap_sentence_rate=1.0):
    global VOCAB_SIZE
    def _generator():
        data_prefix = '/home/user/Desktop/cafa/dump/seqs_'
        max_idx = int(max([float(i.replace('seqs_', '').replace('.pkl', '')) for i in os.listdir('/home/user/Desktop/cafa/dump/') if 'seqs' in i]))
        while True:
            for i in range(0, max_idx, 100000):
                seq_tokens = pickle.load(open(data_prefix + str(i) + '.pkl', 'rb'))
                """
                padded_seq_tokens = []
                for tokens in tqdm(seq_tokens):
                    if len(tokens) <= MAX_SEQ_TOKEN_LEN:
                        padded_seq_tokens += pad_tokens(tokens, MAX_SEQ_TOKEN_LEN, PAD_TOKEN)
                    else:
                        # padded_seq_tokens += pad_tokens(tokens[:MAX_SEQ_TOKEN_LEN/2], MAX_SEQ_TOKEN_LEN, PAD_TOKEN)
                        # padded_seq_tokens += pad_tokens(tokens[-MAX_SEQ_TOKEN_LEN/2:], MAX_SEQ_TOKEN_LEN, PAD_TOKEN)
                        padded_seq_tokens += pad_tokens(tokens[:MAX_SEQ_TOKEN_LEN], MAX_SEQ_TOKEN_LEN, PAD_TOKEN) ## todo - take start and end
                padded_seq_tokens = np.array(padded_seq_tokens, dtype='int8')
                """
                padded_seq_tokens = np.array( [pad_tokens(tokens, MAX_SEQ_TOKEN_LEN, PAD_TOKEN) for tokens in tqdm(seq_tokens) if len(tokens) <= MAX_SEQ_TOKEN_LEN], dtype='int8')
                BATCH_SIZE = 32
                for batch_idx in range(0, padded_seq_tokens.shape[0], BATCH_SIZE):
                    batch =  gen_batch_inputs(
                        [padded_seq_tokens[batch_idx: batch_idx+BATCH_SIZE], padded_seq_tokens[batch_idx: batch_idx+BATCH_SIZE]],
                        token_dict,
                        token_list,
                        seq_len=MAX_SEQ_TOKEN_LEN,
                        mask_rate=mask_rate,
                        swap_sentence_rate=swap_sentence_rate,
                    )

                    # By Sample
                    for sample_idx in range(len(batch[0][0])):

                        yield  [[np.expand_dims(batch[0][0][sample_idx], axis=0), np.expand_dims(batch[0][1][sample_idx], axis=0), np.expand_dims(batch[0][2][sample_idx], axis=0)],
                               [np.expand_dims(batch[1][0][sample_idx], axis=0), np.expand_dims(batch[1][1][sample_idx], axis=0)]]

                    # By Batch
                    # yield [batch[0][sample_idx], batch[1][sample_idx]]

    return _generator


def get_bert_generator_supervised(MAX_SEQ_TOKEN_LEN, PAD_TOKEN):
    global VOCAB_SIZE
    def _generator():
        while True:
            data_prefix = '/home/user/Desktop/cafa/dump/seqs_'
            max_idx = int(max( [float(i.replace('seqs_', '').replace('.pkl', '')) for i in os.listdir('/home/user/Desktop/cafa/dump/') if 'seqs' in i]))
            for i in range(0, max_idx, 100000):
                seq_tokens = pickle.load(open(data_prefix + str(i) + '.pkl', 'rb'))
                padded_seq_tokens = np.array( [pad_tokens(tokens, MAX_SEQ_TOKEN_LEN, PAD_TOKEN) for tokens in tqdm(seq_tokens) if len(tokens) <= MAX_SEQ_TOKEN_LEN], dtype='int8')

                annotations = pickle.load(open('/home/user/Desktop/annots_all.pkl', 'rb'))
                # unique_annotations = sorted(set.union(*map(set, tqdm(annotations, desc='Unique Annots'))))
                # pickle.dump(unique_annotations, open('unique_a.pkl', 'wb'))
                unique_annotations = pickle.load(open('unique_a.pkl', 'rb'))

                n_unique_annotations = len(unique_annotations)
                print('There are %d unique annotations.' % n_unique_annotations)
                annotation_to_index = {annotation: i for i, annotation in enumerate(unique_annotations)}
                encoded_annotations = np.array([encode_labels(record_annotations, annotation_to_index, n_unique_annotations) for record_annotations, record_tokens in zip(annotations, seq_tokens) if len(record_tokens) <= MAX_SEQ_TOKEN_LEN], dtype='int8')

                BATCH_SIZE = 32
                for batch_idx in range(0, padded_seq_tokens.shape[0], BATCH_SIZE):

                    token_input = padded_seq_tokens[batch_idx: batch_idx + BATCH_SIZE]
                    seg_input = np.zeros((token_input.shape[0], token_input.shape[1]))
                    mask_input = np.ones((token_input.shape[0], token_input.shape[1]))

                    batch = [[token_input, seg_input, mask_input],
                             encoded_annotations[batch_idx: batch_idx + BATCH_SIZE]]

                    # By Sample
                    for sample_idx in range(len(batch[0][0])):

                        yield  [[np.expand_dims(batch[0][0][sample_idx], axis=0), np.expand_dims(batch[0][1][sample_idx], axis=0), np.expand_dims(batch[0][2][sample_idx], axis=0)],
                               np.expand_dims(batch[1][sample_idx], axis=0)]

                    # By Batch
                    # yield [batch[0][sample_idx], batch[1][sample_idx]]

    return _generator


def get_bert_generator_old(token_input, token_dict, token_list, seq_len=20, mask_rate=0.3, swap_sentence_rate=1.0):

    def _generator():
        while True:
            # for file in all_files:
            #load pickle
            #yield the pickled data
            yield gen_batch_inputs(
                token_input,
                token_dict,
                token_list,
                seq_len=seq_len,
                mask_rate=mask_rate,
                swap_sentence_rate=swap_sentence_rate,
            )

    return _generator


def gen_token_dict(token_input):
    # Build token dictionary
    token_dict = get_base_dict()  # A dict that contains some special tokens

    for pairs in token_input:
        for token in np.concatenate((pairs[0], pairs[1])):
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    token_list = list(token_dict.keys())  # Used for selecting a random word
    return token_dict, token_list

def identity(x):
    return x

symbolic = identity
if hasattr(K, 'symbolic'):
    symbolic = K.symbolic

class AdamWarmup(keras.optimizers.Optimizer):
    """Adam optimizer with warmup.

    Default parameters follow those provided in the original paper.

    # Arguments
        decay_steps: Learning rate will decay linearly to zero in decay steps.
        warmup_steps: Learning rate will increase linearly to lr in first warmup steps.
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        weight_decay: float >= 0. Weight decay.
        weight_decay_pattern: A list of strings. The substring of weight names to be decayed.
                              All weights will be decayed if it is None.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, weight_decay=0., weight_decay_pattern=None,
                 amsgrad=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_weight_decay = weight_decay
        self.weight_decay_pattern = weight_decay_pattern
        self.amsgrad = amsgrad

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    @symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.min_lr + (self.lr - self.min_lr) * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
        )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_{}'.format(i)) for i, p in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_{}'.format(i)) for i, p in enumerate(params)]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vh_{}'.format(i)) for i, p in enumerate(params)]
        else:
            vhats = [K.zeros(1, dtype=K.dtype(p), name='vh_{}'.format(i)) for i, p in enumerate(params)]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if self.initial_weight_decay > 0.0:
                if self.weight_decay_pattern is None:
                    p_t += self.weight_decay * p
                else:
                    for pattern in self.weight_decay_pattern:
                        if pattern in p.name:
                            p_t += self.weight_decay * p
                            break
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'decay_steps': float(K.get_value(self.decay_steps)),
            'warmup_steps': float(K.get_value(self.warmup_steps)),
            'min_lr': float(K.get_value(self.min_lr)),
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'weight_decay': float(K.get_value(self.weight_decay)),
            'weight_decay_pattern': self.weight_decay_pattern,
            'amsgrad': self.amsgrad,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))