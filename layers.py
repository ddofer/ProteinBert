
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv1D, Multiply, Add

class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class MultiHeadSelfAttention(keras.layers.Layer):

    def __init__(self, n_heads, d_key, d_value, **kwargs):
        self.n_heads = n_heads
        self.d_key = d_key
        self.sqrt_d_key = np.sqrt(self.d_key)
        self.d_value = d_value
        self.d_output = n_heads * d_value
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, length, d_input)
        _, _, self.d_input = input_shape
        self.d_input = int(self.d_input)
        # Wq, Wk: (n_heads, d_input, d_key)
        self.Wq = self.add_weight(name = 'Wq', shape = (self.n_heads, self.d_input, self.d_key), initializer = 'glorot_uniform', trainable = True)
        self.Wk = self.add_weight(name = 'Wk', shape = (self.n_heads, self.d_input, self.d_key), initializer = 'glorot_uniform', trainable = True)
        # Wv: (n_heads, d_input, d_value)
        self.Wv = self.add_weight(name = 'Wv', shape = (self.n_heads, self.d_input, self.d_value), initializer = 'glorot_uniform', trainable = True)
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, X, *args):
        # X: (batch_size, length, d_input)
        _, length, d_input = K.int_shape(X)
        assert d_input == self.d_input
        QX = K.tanh(K.dot(X, self.Wq)) # (batch_size, length, n_heads, d_key)
        # (batch_size * n_heads, length, d_key)
        # random pooling
        QX = K.reshape(K.permute_dimensions(QX, (0, 2, 1, 3)), (-1, length, self.d_key))
        KX = K.tanh(K.dot(X, self.Wk)) # (batch_size, length, n_heads, d_key)
        # (batch_size * n_heads, length, d_key)
        KX = K.reshape(K.permute_dimensions(KX, (0, 2, 1, 3)), (-1, length, self.d_key))
        VX = K.relu(K.dot(X, self.Wv)) # (batch_size, length, n_heads, d_value)
        # (batch_size * n_heads, length, d_value)
        VX = K.reshape(K.permute_dimensions(VX, (0, 2, 1, 3)), (-1, length, self.d_value))
        # (batch_size * n_heads, length, length)
        Z = K.softmax(K.batch_dot(QX, K.permute_dimensions(KX, (0, 2, 1))) / self.sqrt_d_key)
        Y = K.batch_dot(Z, VX) # (batch_size * n_heads, length, d_value)
        # (batch_size, length, n_heads, d_value)
        Y = K.permute_dimensions(K.reshape(Y, (-1, self.n_heads, length, self.d_value)), (0, 2, 1, 3))
        # (batch_size, length, n_heads * d_value)
        return K.reshape(Y, (-1, length, self.d_output))

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, length, d_input)
        batch_size, length, _ = input_shape
        return (batch_size, length, self.d_output)


class TransformerBlock(keras.layers.Layer):

    def __init__(self, n_heads, d_seq, d_key, d_vec, dense_activation = 'relu', **kwargs):
        assert d_seq % n_heads == 0
        self.n_heads = n_heads
        self.d_seq = d_seq
        self.d_key = d_key
        self.d_vec = d_vec
        self.d_value = d_seq // n_heads
        name = kwargs.get('name', 'transformer')
        self.attention = MultiHeadSelfAttention(self.n_heads, self.d_key, self.d_value, name = '%s-attention' % name)
        self.attention_norm = LayerNormalization(name = '%s-attention-norm' % name)
        self.seq_dense1 = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-seq-dense1' % name)
        self.seq_norm1 = LayerNormalization(name = '%s-seq-norm1' % name)
        self.vec_dense1 = keras.layers.Dense(self.d_vec, activation = dense_activation, name = '%s-vec-dense1' % name)
        self.seq_mean_dense = keras.layers.Dense(self.d_vec, activation = dense_activation, name = '%s-seq-mean-dense' % name)
        self.vec_norm1 = LayerNormalization(name = '%s-vec-norm1' % name)
        self.vec_dense2 = keras.layers.Dense(self.d_vec, activation = dense_activation, name = '%s-vec-dense2' % name)
        self.vec_norm2 = LayerNormalization(name = '%s-vec-norm2' % name)
        self.vec_seqing_dense = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-vec-seqing-dense' % name)
        self.seq_dense2 = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-seq-dense2' % name)
        self.seq_norm2 = LayerNormalization(name = '%s-seq-norm2' % name)
        self.seq_dense3 = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-seq-dense3' % name)
        self.seq_norm3 = LayerNormalization(name = '%s-seq-norm3' % name)
        self.layers_with_seq_input = [self.attention, self.attention_norm, self.seq_dense1, self.seq_norm1, self.seq_dense2, self.seq_norm2, self.seq_dense3, self.seq_norm3]
        self.layers_with_vec_input = [self.vec_dense1, self.vec_norm1, self.vec_dense2, self.vec_norm2, self.vec_seqing_dense]
        self.all_layers = self.layers_with_seq_input + self.layers_with_vec_input + [self.seq_mean_dense]
        super(TransformerBlock, self).__init__(**kwargs)

    def build(self, input_shapes):
        seq_shape, vec_shape = input_shapes
        batch_size, length, d_seq = seq_shape
        batch_size2, d_vec = vec_shape
        assert d_seq == self.d_seq
        assert d_vec == self.d_vec
        for layer in self.layers_with_seq_input: layer.build(seq_shape)
        for layer in self.layers_with_vec_input: layer.build(vec_shape)
        self.seq_mean_dense.build((batch_size, d_seq))
        self._trainable_weights = [weight for layer in self.all_layers for weight in layer._trainable_weights]
        super(TransformerBlock, self).build([seq_shape, vec_shape])

    def compute_output_shape(self, input_shapes):
        return input_shapes

    def call(self, X, *args):
        X_seq, X_vec = X
        X_seq = self.attention_norm(keras.layers.Add()([X_seq, self.attention(X_seq)]))
        X_seq = self.seq_norm1(keras.layers.Add()([X_seq, self.seq_dense1(X_seq)]))
        # (batch_size, length, d_seq) --> (batch_size, d_seq)
        X_seq_mean = K.mean(X_seq, axis = 1)
        X_vec = self.vec_norm1(keras.layers.Add()([X_vec, self.vec_dense1(X_vec), self.seq_mean_dense(X_seq_mean)]))
        X_vec = self.vec_norm2(keras.layers.Add()([X_vec, self.vec_dense2(X_vec)]))
        # (batch_size, d_vec) --> (batch_size, 1, d_seq)
        X_vec_seqed = K.expand_dims(self.vec_seqing_dense(X_vec), axis = 1)
        X_seq = self.seq_norm2(keras.layers.Add()([X_seq, self.seq_dense2(X_seq), X_vec_seqed]))
        X_seq = self.seq_norm3(keras.layers.Add()([X_seq, self.seq_dense3(X_seq)]))
        return [X_seq, X_vec]


class Transformer(keras.layers.Layer):

    def __init__(self, n_heads, d_seq, d_key, dense_activation='relu', **kwargs):
        assert d_seq % n_heads == 0
        self.n_heads = n_heads
        self.d_seq = d_seq
        self.d_key = d_key
        self.d_value = d_seq // n_heads
        self.attention = MultiHeadSelfAttention(self.n_heads, self.d_key, self.d_value)
        self.attention_norm = LayerNormalization()
        self.seq_dense1 = keras.layers.Dense(self.d_seq, activation=dense_activation)
        self.seq_norm1 = LayerNormalization()
        self.layers_with_seq_input = [self.attention, self.attention_norm, self.seq_dense1, self.seq_norm1]
        self.all_layers = self.layers_with_seq_input
        super(Transformer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        return input_shapes

    def call(self, X, *args):
        X_seq = X
        X_seq = self.attention_norm(keras.layers.Add()([X_seq, self.attention(X_seq)]))
        X_seq = self.seq_norm1(keras.layers.Add()([X_seq, self.seq_dense1(X_seq)]))
        # (batch_size, length, d_seq) --> (batch_size, d_seq)
        return X_seq


class MessagePassing_SeqToState(keras.layers.Layer):
    def __init__(self, d_vec, dense_activation='relu'):
        self.seq_mean_dense = keras.layers.Dense(d_vec, activation=dense_activation)
        self.vec_dense1 = keras.layers.Dense(d_vec, activation=dense_activation)
        self.vec_dense2 = keras.layers.Dense(d_vec, activation=dense_activation)
        self.vec_norm1 = LayerNormalization()
        self.vec_norm2 = LayerNormalization()
        super(MessagePassing_SeqToState, self).__init__()

    def compute_output_shape(self, input_shapes):
        return input_shapes[1]

    def call(self, X, *args):
        X_seq, X_vec = X
        X_seq_mean = K.mean(X_seq, axis = 1)
        X_vec = self.vec_norm1(keras.layers.Add()([X_vec, self.vec_dense1(X_vec), self.seq_mean_dense(X_seq_mean)]))
        X_vec = self.vec_norm2(keras.layers.Add()([X_vec, self.vec_dense2(X_vec)]))
        return X_vec


class MessagePassing_StateToSeq(keras.layers.Layer):
    def __init__(self, d_seq, dense_activation='relu'):
        self.vec_seqing_dense = keras.layers.Dense(d_seq, activation=dense_activation)
        self.seq_dense2 = keras.layers.Dense(d_seq, activation=dense_activation)
        self.seq_norm2 = LayerNormalization()
        self.seq_dense3 = keras.layers.Dense(d_seq, activation=dense_activation)
        self.seq_norm3 = LayerNormalization()
        super(MessagePassing_StateToSeq, self).__init__()

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, X, *args):
        X_seq, X_vec = X
        # (batch_size, d_vec) --> (batch_size, 1, d_seq)
        X_vec_seqed = K.expand_dims(self.vec_seqing_dense(X_vec), axis=1)
        X_seq = self.seq_norm2(keras.layers.Add()([X_seq, self.seq_dense2(X_seq), X_vec_seqed]))
        X_seq = self.seq_norm3(keras.layers.Add()([X_seq, self.seq_dense3(X_seq)]))
        return X_seq


def get_sinusoidal_embedding(d_pos_vec, n_position):
    position_enc = np.array(
        [[pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)] if pos != 0 else np.zeros(d_pos_vec) for
         pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc


class SeqInputEmbedding(keras.layers.Layer):

    def __init__(self, vocab_size, d_positional_embedding, **kwargs):
        self.vocab_size = vocab_size
        self.d_positional_embedding = d_positional_embedding
        self.d_total = self.vocab_size + self.d_positional_embedding
        super(SeqInputEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, length)
        _, self.length = input_shape
        # (length, d_positional_embedding)
        self.position_embeddings = K.constant(get_sinusoidal_embedding(self.d_positional_embedding, self.length))
        super(SeqInputEmbedding, self).build(input_shape)

    def call(self, X, *args):
        # X is the input tokens, given as integers of shape (batch_size, length)
        _, length = K.int_shape(X)
        assert length == self.length
        batch_size = K.shape(X)[0]
        # (batch_size, length, vocab_size)
        token_embedding = K.one_hot(X, self.vocab_size)
        # (batch_size, length, d_positional_embedding)
        positional_embedding = K.tile(K.reshape(self.position_embeddings, (1, length, self.d_positional_embedding)),
                                      (batch_size, 1, 1))
        # (batch_size, length, d_total)
        return K.concatenate([token_embedding, positional_embedding])

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, length)
        batch_size, length = input_shape
        return (batch_size, length, self.d_total)


class OneHotEncoding(keras.layers.Layer):

    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        super(OneHotEncoding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.vocab_size,)

    def call(self, X, *args):
        return K.one_hot(X, self.vocab_size)


class GlobalAttention(keras.layers.Layer):
    '''
    Recevies two inputs:
    1. A global representation (of some fixed dimension)
    2. A sequence (of any length, and some fixed dimension)
    The global representation is used to construct a global query that attends to all the positions in the sequence (independently
    for any of the heads).
    '''

    def __init__(self, n_heads, d_key, d_value, **kwargs):
        self.n_heads = n_heads
        self.d_key = d_key
        self.sqrt_d_key = np.sqrt(self.d_key)
        self.d_value = d_value
        self.d_output = n_heads * d_value
        super(GlobalAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        print('compute_output_shape', input_shapes)  # XXX
        (batch_size, _), _ = input_shapes
        return (batch_size, self.d_output)

    def build(self, input_shapes):
        # input_shapes: (batch_size, d_global_input), (batch_size, length, d_seq_input)
        (_, self.d_global_input), (_, _, self.d_seq_input) = input_shapes
        # Wq: (n_heads, d_global_input, d_key)
        self.Wq = self.add_weight(name='Wq', shape=(self.n_heads, self.d_global_input, self.d_key),
                                  initializer='glorot_uniform', trainable=True)
        # Wk: (n_heads, d_seq_input, d_key)
        self.Wk = self.add_weight(name='Wk', shape=(self.n_heads, self.d_seq_input, self.d_key),
                                  initializer='glorot_uniform', trainable=True)
        # Wv: (n_heads, d_seq_input, d_value)
        self.Wv = self.add_weight(name='Wv', shape=(self.n_heads, self.d_seq_input, self.d_value),
                                  initializer='glorot_uniform', trainable=True)
        super(GlobalAttention, self).build(input_shapes)

    def call(self, inputs, *args):
        # X: (batch_size, d_global_input)
        # S: (batch_size, length, d_seq_input)
        X, S = inputs
        _, length, _ = K.int_shape(S)
        # (batch_size, n_heads, d_key)
        QX = K.tanh(K.dot(X, self.Wq))
        # (batch_size * n_heads, d_key)
        QX_batched_heads = K.reshape(QX, (-1, self.d_key))
        # (batch_size, n_heads, d_key, length)
        KS = K.permute_dimensions(K.tanh(K.dot(S, self.Wk)), (0, 2, 3, 1))
        # (batch_size * n_heads, d_key, length)
        KS_batched_heads = K.reshape(KS, (-1, self.d_key, length))
        # (batch_size, n_heads, length, d_value)
        VS = K.permute_dimensions(K.relu(K.dot(S, self.Wv)), (0, 2, 1, 3))
        # (batch_size * n_heads, length, d_value)
        VS_batched_heads = K.reshape(VS, (-1, length, self.d_value))
        # (batch_size * n_heads, length)
        Z_batched_heads = K.softmax(K.batch_dot(QX_batched_heads, KS_batched_heads) / self.sqrt_d_key)
        # (batch_size * n_heads, d_value)
        Y_batched_heads = K.batch_dot(Z_batched_heads, VS_batched_heads)
        # (batch_size, n_heads * d_value)
        Y = K.reshape(Y_batched_heads, (-1, self.d_output))
        return Y


class MultiHeadSelfAttention2(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=6):
        super(MultiHeadSelfAttention2, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = keras.layers.Dense(embed_dim)
        self.key_dense = keras.layers.Dense(embed_dim)
        self.value_dense = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads( value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose( attention, perm=[0, 2, 1, 3] )
        concat_attention = tf.reshape( attention, (batch_size, -1, self.embed_dim) )
        output = self.combine_heads( concat_attention )
        return output


class TransformerBlock2(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock2, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential( [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),] )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, *args):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x, *args):
        maxlen = K.int_shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
