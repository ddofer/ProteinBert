
import os
import re
import sys
import json
import keras
import random
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras_bert.keras_bert.bert import get_model
from keras_bert import gen_batch_inputs, get_base_dict, compile_model
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda
from keras_bert.loader import load_trained_model_from_checkpoint, build_model_from_config

from bert_stuff import *


def train_weaponized_bert(model, token_dict, train_labels, batch_size=32, epochs=10):

    lr = 2e-5
    weight_decay = 0.001
    nb_epochs = epochs 
    bsz = batch_size
    num_of_all_samples = 100000000
    decay_steps = int(nb_epochs * num_of_all_samples / bsz)
    warmup_steps = int(0.1 * decay_steps)
    adamwarm = AdamWarmup(lr=lr, decay_steps=decay_steps, warmup_steps=warmup_steps, weight_decay=weight_decay)

    sequence_output = model.layers[-6].output
    pool_output = Dense(train_labels.shape[1], activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02), name='real_output')(sequence_output)
    model3 = Model(inputs=model.input, outputs=pool_output)
    model3.compile(loss='binary_crossentropy', optimizer=adamwarm)
    model3.summary()

    # seg_input = np.zeros((token_input.shape[0], token_input.shape[1]))
    # mask_input = np.ones((token_input.shape[0], token_input.shape[1]))
    # model3.fit([token_input, seg_input, mask_input], train_labels, batch_size=bsz, epochs=nb_epochs)
    return model3


def exract_embeddings(model, token_dict, token_input, train_labels):
    sequence_output = model.layers[-6].output
    model3 = Model(inputs=model.input, outputs=sequence_output)
    seg_input = np.zeros((token_input.shape[0], token_input.shape[1]))
    mask_input = np.ones((token_input.shape[0], token_input.shape[1]))
    return model3.predict([token_input, seg_input, mask_input])


def exract_prediction(model, token_dict, token_input, train_labels):
    seg_input = np.zeros((token_input.shape[0], token_input.shape[1]))
    mask_input = np.ones((token_input.shape[0], token_input.shape[1]))
    return model.predict([token_input, seg_input, mask_input])


def unsupervised_weaponized_bert(token_dict, annotations, MAX_SEQ_TOKEN_LEN, PAD_TOKEN, batch_size=32, epochs=1000000):


    model = get_model(
        token_num=len(token_dict),
        head_num=5,
        transformer_num=12,
        embed_dim=25,
        feed_forward_dim=100,
        seq_len=None, # token_input[0].shape[1],
        pos_num=MAX_SEQ_TOKEN_LEN,
        dropout_rate=0.05,
    )

    model.compile( optimizer=AdamWarmup( decay_steps=100000, warmup_steps=10000, lr=2e-5, weight_decay=0.01, weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo']), loss=keras.losses.sparse_categorical_crossentropy)
    model.summary()

    generator = get_bert_generator(token_dict, annotations, list(token_dict.keys()), MAX_SEQ_TOKEN_LEN, PAD_TOKEN, seq_len=MAX_SEQ_TOKEN_LEN, mask_rate=0.3, swap_sentence_rate=1.0)
    # plot_model(model, 'bert.png')

    """
    model.fit_generator(
        generator=generator(),
        steps_per_epoch=101800000, # Fancy Batch size
        epochs=epochs,
        validation_data=generator(),
        validation_steps=100)
        # callbacks=[ keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) ],)
    """
    return model, token_dict
