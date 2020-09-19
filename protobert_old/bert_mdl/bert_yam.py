

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
from keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda


sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

lr = 2e-5
weight_decay = 0.001
nb_epochs = 1
bsz = 32
maxlen = 72

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
train_df = train_df.sample(frac=0.001, random_state=42)
train_lines, train_labels = train_df['comment_text'].values, train_df.target.values

decay_steps = int(nb_epochs * train_lines.shape[0] / bsz)
warmup_steps = int(0.1 * decay_steps)

from keras_bert.keras_bert.bert import get_model
from keras_bert.keras_bert.loader import load_trained_model_from_checkpoint, build_model_from_config

print('begin_build')

config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

bert_config = """
{
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 1024, 
  "initializer_range": 0.02, 
  "intermediate_size": 4096, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 16, 
  "num_hidden_layers": 24, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": 28996
}
"""

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
            self.lr = K.variable(lr, name='lr')
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
            self.lr * (t / self.warmup_steps),
            self.lr * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
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
pool_output = Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02), name='real_output')(sequence_output)
model3 = Model(inputs=model.input, outputs=pool_output)
model3.compile(loss='binary_crossentropy', optimizer=adamwarm)
model3.summary()

names = [weight.name for layer in model3.layers for weight in layer.weights]
weights = model3.get_weights()

for name, weight in zip(names, weights):
    print(name, weight.shape)

    bert_config = """
    {
      "attention_probs_dropout_prob": 0.1, 
      "directionality": "bidi", 
      "hidden_act": "gelu", 
      "hidden_dropout_prob": 0.1, 
      "hidden_size": 1024, 
      "initializer_range": 0.02, 
      "intermediate_size": 4096, 
      "max_position_embeddings": %s, 
      "num_attention_heads": 16, 
      "num_hidden_layers": 24, 
      "pooler_fc_size": 768, 
      "pooler_num_attention_heads": 12, 
      "pooler_num_fc_layers": 3, 
      "pooler_size_per_head": 128, 
      "pooler_type": "first_token_transform", 
      "type_vocab_size": 2, 
      "vocab_size": 16000
    }
    """ % (token_input.shape[1], )




def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
        tokens_a = tokenizer.tokenize(example[i])
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (
                    max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)


dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
print('build tokenizer done')
print('sample used', train_lines.shape)
token_input = convert_lines(train_lines, maxlen, tokenizer)
seg_input = np.zeros((token_input.shape[0], maxlen))
mask_input = np.ones((token_input.shape[0], maxlen))
print(token_input.shape)
print(seg_input.shape)
print(mask_input.shape)
print('begin training')
model3.fit([token_input, seg_input, mask_input], train_labels, batch_size=bsz, epochs=nb_epochs)

# %% [code]
# load test data
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
# test_df['comment_text'] = test_df['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
eval_lines = test_df['comment_text'].values
print(eval_lines.shape)
print('load data done')
token_input2 = convert_lines(eval_lines, maxlen, tokenizer)
seg_input2 = np.zeros((token_input2.shape[0], maxlen))
mask_input2 = np.ones((token_input2.shape[0], maxlen))
print('test data done')
print(token_input2.shape)
print(seg_input2.shape)
print(mask_input2.shape)
hehe = model3.predict([token_input2, seg_input2, mask_input2], verbose=1, batch_size=bsz)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv',
                         index_col='id')
submission['prediction'] = hehe
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)