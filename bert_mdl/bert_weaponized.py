
import os
import re
import sys
import json
import keras
import random
import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras import backend as K
from keras_bert.keras_bert.bert import get_model
from keras_bert import gen_batch_inputs, get_base_dict, compile_model
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda
from keras_bert.loader import load_trained_model_from_checkpoint, build_model_from_config

def train_weaponized_bert(token_input, train_labels, batch_size=32, epochs=1000000, model=None):

    # sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
    # BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
    # print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

    lr = 2e-5
    weight_decay = 0.001
    nb_epochs = epochs 
    bsz = batch_size 
    maxlen = token_input.shape[1]

    train_lines = token_input.shape[0]
    decay_steps = int(nb_epochs * train_lines / bsz)
    warmup_steps = int(0.1 * decay_steps)

    print('Building')

    if model is None:
        bert_config = """
        {
          "attention_probs_dropout_prob": 0.1, 
          "directionality": "bidi", 
          "hidden_act": "gelu", 
          "hidden_dropout_prob": 0.1, 
          "hidden_size": 128, 
          "initializer_range": 0.02, 
          "intermediate_size": 128, 
          "max_position_embeddings": %s, 
          "num_attention_heads": 16, 
          "num_hidden_layers": 24, 
          "pooler_fc_size": 128, 
          "pooler_num_attention_heads": 12, 
          "pooler_num_fc_layers": 3, 
          "pooler_size_per_head": 128, 
          "pooler_type": "first_token_transform", 
          "type_vocab_size": 2, 
          "vocab_size": 16000
        }
        """ % (token_input.shape[1], )

        open('temp', 'w').write(bert_config)
        model, _ = build_model_from_config('temp', training=True)

    model.summary(line_length=120)

    class AdamWarmup(keras.optimizers.Optimizer):
        def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                     lr=0.001, beta_1=0.9, beta_2=0.999,
                     epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                     amsgrad=False, **kwargs):
            super(AdamWarmup, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.decay_steps = K.variable(decay_steps, name='decay_steps')
                self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
                self.min_lr = K.variable(min_lr, name='min_lr')
                self.iterations = K.variable(0, dtype='int64', name='iterations')
                self.learning_rate = K.variable(lr, name='lr')
                self.beta_1 = K.variable(beta_1, name='beta_1')
                self.beta_2 = K.variable(beta_2, name='beta_2')
                self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
                self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
            if epsilon is None:
                epsilon = K.epsilon()
            self.epsilon = epsilon
            self.initial_kernel_weight_decay = kernel_weight_decay
            self.initial_bias_weight_decay = bias_weight_decay
            self.amsgrad = amsgrad

        def get_updates(self, loss, params):
            grads = self.get_gradients(loss, params)
            self.updates = [K.update_add(self.iterations, 1)]

            t = K.cast(self.iterations, K.floatx()) + 1

            lr = K.switch(
                t <= self.warmup_steps,
                self.learning_rate * (t / self.warmup_steps),
                self.learning_rate * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
                )

            lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                         (1. - K.pow(self.beta_1, t)))

            ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            if self.amsgrad:
                vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            else:
                vhats = [K.zeros(1) for _ in params]
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

                if 'bias' in p.name or 'Norm' in p.name:
                    if self.initial_bias_weight_decay > 0.0:
                        p_t += self.bias_weight_decay * p
                else:
                    if self.initial_kernel_weight_decay > 0.0:
                        p_t += self.kernel_weight_decay * p
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
                'lr': float(K.get_value(self.lr)),
                'beta_1': float(K.get_value(self.beta_1)),
                'beta_2': float(K.get_value(self.beta_2)),
                'epsilon': self.epsilon,
                'kernel_weight_decay': float(K.get_value(self.kernel_weight_decay)),
                'bias_weight_decay': float(K.get_value(self.bias_weight_decay)),
                'amsgrad': self.amsgrad,
            }
            base_config = super(AdamWarmup, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    adamwarm = AdamWarmup(lr=lr, decay_steps=decay_steps, warmup_steps=warmup_steps, kernel_weight_decay=weight_decay)
    sequence_output = model.layers[-6].output
    pool_output = Dense(train_labels.shape[1], activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02), name='real_output')(sequence_output)
    model3 = Model(inputs=model.input, outputs=pool_output)
    model3.compile(loss='binary_crossentropy', optimizer=adamwarm)
    model3.summary()

    names = [weight.name for layer in model3.layers for weight in layer.weights]
    weights = model3.get_weights()

    for name, weight in zip(names, weights): print(name, weight.shape)


    print('build tokenizer done')

    seg_input = np.zeros((token_input.shape[0], maxlen))
    mask_input = np.ones((token_input.shape[0], maxlen))
    print(token_input.shape)
    print(seg_input.shape)
    print(mask_input.shape)
    print('begin training')
    print(train_labels.shape)
    model3.fit([token_input, seg_input, mask_input], train_labels, batch_size=bsz, epochs=nb_epochs)
    return model3


def unsupervised_weaponized_bert(token_input, batch_size=32, epochs=1000000):

    lr = 2e-5
    nb_epochs = epochs
    bsz = batch_size
    train_lines = token_input.shape[0]
    token_input = np.asarray(token_input, dtype='str')
    token_input = [token_input, token_input]
    class AdamWarmup(keras.optimizers.Optimizer):
        def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                     lr=0.001, beta_1=0.9, beta_2=0.999,
                     epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                     amsgrad=False, **kwargs):
            super(AdamWarmup, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.decay_steps = K.variable(decay_steps, name='decay_steps')
                self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
                self.min_lr = K.variable(min_lr, name='min_lr')
                self.iterations = K.variable(0, dtype='int64', name='iterations')
                self.learning_rate = K.variable(lr, name='lr')
                self.beta_1 = K.variable(beta_1, name='beta_1')
                self.beta_2 = K.variable(beta_2, name='beta_2')
                self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
                self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
            if epsilon is None:
                epsilon = K.epsilon()
            self.epsilon = epsilon
            self.initial_kernel_weight_decay = kernel_weight_decay
            self.initial_bias_weight_decay = bias_weight_decay
            self.amsgrad = amsgrad

        def get_updates(self, loss, params):
            grads = self.get_gradients(loss, params)
            self.updates = [K.update_add(self.iterations, 1)]

            t = K.cast(self.iterations, K.floatx()) + 1

            lr = K.switch(
                t <= self.warmup_steps,
                self.learning_rate * (t / self.warmup_steps),
                self.learning_rate * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
                )

            lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                         (1. - K.pow(self.beta_1, t)))

            ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            if self.amsgrad:
                vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            else:
                vhats = [K.zeros(1) for _ in params]
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

                if 'bias' in p.name or 'Norm' in p.name:
                    if self.initial_bias_weight_decay > 0.0:
                        p_t += self.bias_weight_decay * p
                else:
                    if self.initial_kernel_weight_decay > 0.0:
                        p_t += self.kernel_weight_decay * p
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
                'lr': float(K.get_value(self.lr)),
                'beta_1': float(K.get_value(self.beta_1)),
                'beta_2': float(K.get_value(self.beta_2)),
                'epsilon': self.epsilon,
                'kernel_weight_decay': float(K.get_value(self.kernel_weight_decay)),
                'bias_weight_decay': float(K.get_value(self.bias_weight_decay)),
                'amsgrad': self.amsgrad,
            }
            base_config = super(AdamWarmup, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Build token dictionary
    token_dict = get_base_dict()  # A dict that contains some special tokens
    for pairs in token_input:
        for token in pairs[0] + pairs[1]:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    token_list = list(token_dict.keys())  # Used for selecting a random word

    # Build & train the model
    model = get_model(
        token_num=len(token_dict),
        head_num=5,
        transformer_num=12,
        embed_dim=25,
        feed_forward_dim=100,
        seq_len=20,
        pos_num=20,
        dropout_rate=0.05,
    )

    model.compile( optimizer=AdamWarmup( decay_steps=100000, warmup_steps=10000, lr=lr, kernel_weight_decay=0.01, bias_weight_decay=0.01), loss=keras.losses.sparse_categorical_crossentropy) # , weight_decay=0.01, weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'], ), loss=keras.losses.sparse_categorical_crossentropy,)
    model.summary()

    # sentence_pairs: list of lists..

    def _generator():
        while True:
            yield gen_batch_inputs(
                [token_input, token_input],
                token_dict,
                token_list,
                seq_len=20,
                mask_rate=0.3,
                swap_sentence_rate=1.0,
            )
    print('FUCK! this actually works!!!!')
    print('FUCK! this actually works!!!!')
    print('FUCK! this actually works!!!!')
    print('FUCK! this actually works!!!!')
    print('FUCK! this actually works!!!!')
    
    
    model.fit_generator(
        generator=_generator(),
        steps_per_epoch=1000,
        epochs=100,
        validation_data=_generator(),
        validation_steps=100,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ],)
    return model

