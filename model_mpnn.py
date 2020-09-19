
import keras
import numpy as np
from keras import backend as K
from keras.layers import Conv1D, Multiply, Add

from layers import *

def transformer_block(x, n_heads, d_seq, d_key, d_vec, dense_activation = 'relu', **kwargs):
    X_seq, X_vec = x

    X_seq = Transformer(n_heads, d_seq, d_key, dense_activation)(X_seq)
    X_vec = MessagePassing_SeqToState(d_vec, dense_activation)([X_seq, X_vec])
    X_seq = MessagePassing_StateToSeq(d_seq, dense_activation)([X_seq, X_vec])

    return [X_seq, X_vec]


def protobert_transformer(x, vocab_size, d_vec, n_transformers=8, n_heads=8, d_hidden_seq=512, d_key=64, d_hidden_vec=512, d_positional_embedding=16, hidden_activation='relu', output_vec_activation=None, output_vec=True):
    X_seq, X_vec = x
    X_seq = keras.layers.Dense(d_hidden_seq, activation=hidden_activation)(SeqInputEmbedding(vocab_size=vocab_size, d_positional_embedding=d_positional_embedding)(X_seq))
    X_vec = keras.layers.Dense(d_hidden_vec, activation=hidden_activation)(X_vec)
    for idx in range(n_transformers):
        X_seq, X_vec = transformer_block([X_seq, X_vec], n_heads, d_hidden_seq, d_key, d_hidden_vec, dense_activation = hidden_activation)
    X_seq = keras.layers.Dense(vocab_size, activation='softmax')(X_seq)
    X_vec = keras.layers.Dense(d_vec, activation=output_vec_activation)(X_vec)
    return [X_seq, X_vec]


def create_model(max_seq_len, vocab_size, n_annotations):
    input_seq_layer = keras.layers.Input(shape = (max_seq_len,), dtype = np.int32, name = 'input-seq')
    input_annoatations_layer = keras.layers.Input(shape = (n_annotations,), dtype = np.float32, name = 'input-annotations')
    output_seq_layer, output_annoatations_layer = protobert_transformer(x=[input_seq_layer, input_annoatations_layer], vocab_size=vocab_size, d_vec=n_annotations, output_vec_activation='sigmoid')
    output_seq_layer = keras.layers.Reshape(output_seq_layer.shape[1:], name = 'output_seq_layer')(output_seq_layer)
    output_annoatations_layer = keras.layers.Reshape(output_annoatations_layer.shape[1:], name = 'output_annoatations_layer')(output_annoatations_layer)
    return keras.models.Model(inputs = [input_seq_layer, input_annoatations_layer], outputs = [output_seq_layer, output_annoatations_layer])

