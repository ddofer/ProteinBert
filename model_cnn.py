
import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv1D, Multiply, Add

from layers import *

def _create_model(x, seq_len, vocab_size, n_annotations, d_hidden_seq=512, d_hidden_global=2048, n_blocks=6, n_heads=8, d_key=64, conv_kernel_size=9, wide_conv_dilation_rate=5):
    input_seq, input_annoatations = x
    assert d_hidden_global % n_heads == 0
    d_value = d_hidden_global // n_heads
    encoded_input_seq = OneHotEncoding(vocab_size, name='input-seq-encoding')(input_seq)
    hidden_seq = keras.layers.Dense(d_hidden_seq, activation='relu', name='dense-seq-input')(encoded_input_seq)
    hidden_global = keras.layers.Dense(d_hidden_global, activation='relu', name='dense-global-input')(input_annoatations)
    for block_index in range(1, n_blocks + 1):
        seqed_global = keras.layers.Dense(d_hidden_seq, activation='relu',name='global-to-seq-dense-block%d' % block_index)(hidden_global)
        seqed_global = keras.layers.Reshape((1, d_hidden_seq), name='global-to-seq-reshape-block%d' % block_index)(seqed_global)
        narrow_conv_seq = keras.layers.Conv1D(filters=d_hidden_seq, kernel_size=conv_kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu', name='narrow-conv-block%d' % block_index)(hidden_seq)
        wide_conv_seq = keras.layers.Conv1D(filters=d_hidden_seq, kernel_size=conv_kernel_size, strides=1, padding='same', dilation_rate=wide_conv_dilation_rate, activation='relu', name='wide-conv-block%d' % block_index)(hidden_seq)
        hidden_seq = keras.layers.Add(name='seq-merge1-block%d' % block_index)([hidden_seq, seqed_global, narrow_conv_seq, wide_conv_seq])
        hidden_seq = LayerNormalization(name='seq-merge1-norm-block%d' % block_index)(hidden_seq)
        dense_seq = keras.layers.Dense(d_hidden_seq, activation='relu', name='seq-dense-block%d' % block_index)(hidden_seq)
        hidden_seq = keras.layers.Add(name='seq-merge2-block%d' % block_index)([hidden_seq, dense_seq])
        hidden_seq = LayerNormalization(name='seq-merge2-norm-block%d' % block_index)(hidden_seq)
        dense_global = keras.layers.Dense(d_hidden_global, activation='relu', name='global-dense1-block%d' % block_index)(hidden_global)
        attention = GlobalAttention(n_heads, d_key, d_value, name='global-attention-block%d' % block_index)([hidden_global, hidden_seq])
        hidden_global = keras.layers.Add(name='global-merge1-block%d' % block_index)([hidden_global, dense_global, attention])
        hidden_global = LayerNormalization(name='global-merge1-norm-block%d' % block_index)(hidden_global)
        dense_global = keras.layers.Dense(d_hidden_global, activation='relu', name='global-dense2-block%d' % block_index)(hidden_global)
        hidden_global = keras.layers.Add(name='global-merge2-block%d' % block_index)([hidden_global, dense_global])
        hidden_global = LayerNormalization(name='global-merge2-norm-block%d' % block_index)(hidden_global)
    output_seq = keras.layers.Dense(vocab_size, activation='softmax', name='output-seq')(hidden_seq)
    output_annotations = keras.layers.Dense(n_annotations, activation='sigmoid', name='output-annotations')(hidden_global)
    return output_seq, output_annotations


def create_model(max_seq_len, vocab_size, n_annotations):
    input_seq_layer = keras.layers.Input(shape = (max_seq_len,), dtype = np.int32, name = 'input-seq')
    input_annoatations_layer = keras.layers.Input(shape = (n_annotations,), dtype = np.float32, name = 'input-annotations')
    output_seq_layer, output_annoatations_layer = _create_model(x=[input_seq_layer, input_annoatations_layer], vocab_size=vocab_size, seq_len=max_seq_len, n_annotations = n_annotations)
    output_seq_layer = keras.layers.Reshape(output_seq_layer.shape[1:], name = 'output_seq_layer')(output_seq_layer)
    output_annoatations_layer = keras.layers.Reshape(output_annoatations_layer.shape[1:], name = 'output_annoatations_layer')(output_annoatations_layer)
    return keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = [output_seq_layer, output_annoatations_layer])
