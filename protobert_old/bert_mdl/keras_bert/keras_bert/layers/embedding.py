
import numpy as np
import tensorflow as tf
from .backend import keras
from .backend import backend as K
from .keras_pos_embd import PositionEmbedding
from .keras_layer_normalization import LayerNormalization


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), (self.input_dim, self.output_dim)]

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs), K.identity(self.embeddings)]

class PositinalEncoding1(keras.layers.Layer):
    def __init__(self, max_steps=200, max_dims=200, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1: max_dims += 1
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, emb_dim):
        self.vocab_size = position
        self.emb_dim = emb_dim
        super(PositionalEncoding, self).__init__()

    def get_angle(self, position, i, d_model):
        return position / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))


    def positional_encode(self, position, d_model):
        angles = self.get_angle(
            position=tf.expand_dims(tf.range(position, dtype=tf.float32), 1),
            i=tf.expand_dims(tf.range(d_model, dtype=tf.float32), 0),
            d_model=d_model)

        sins = tf.math.sin(angles[:, 0::2])
        coss = tf.math.cos(angles[:, 1::2])

        pos_enc = tf.reshape(
            tf.concat([sins, coss], axis=-1),
            [tf.shape(sins)[0], -1])
        return tf.cast(tf.expand_dims(pos_enc, 0), tf.float32)


    def call(self, inputs):
        pos_encoding = self.positional_encode(inputs.shape[1], inputs.shape[1])
        return inputs + pos_encoding[:, :tf.shape(inputs)[1], :]


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config):
        super(EncoderLayer, self).__init__(name='EncoderLayer')

        self.config = config
        self.vocab_size = config['vocab_size']
        self.embed_dim = self.config['embed_dim'] # d_model
        self.head_num = self.config['head_num'] # h # split_embed_dim * head_num == embed_dim
        self.split_embed_dim = self.config['split_embed_dim'] # dim_k, dim_v # self-attention에는 context vector에 쓰였던 context vector를 위한 attention dim 개념이 없음, 자기 차원끼리 attention을 구하니까 attention을 위한 벡터가 따로 필요없거든
        self.layer_num = config['layer_num'] # PRETRAINING_N
        self.feed_forward_dim = config['feed_forward_dim'] # dim_ffc

        # Define your layers here.
        self.embed = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, input_shape=(None,))

        # Multi Head Attention
        self.mha = MultiHeadAttention(self.config)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.position_wise_fc_1 = tf.keras.layers.Dense(units=self.feed_forward_dim, activation='relu')
        self.position_wise_fc_2 = tf.keras.layers.Dense(units=self.embed_dim)

    def add_positional_encoding(self, embed):

        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(position, d_model):
            angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(d_model)[np.newaxis, :],
                                    d_model)

            # apply sin to even indices in the array; 2i
            sines = np.sin(angle_rads[:, 0::2])
            # apply cos to odd indices in the array; 2i+1
            cosines = np.cos(angle_rads[:, 1::2])
            pos_encoding = np.concatenate([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        pos_encoding = positional_encoding(self.vocab_size, self.embed_dim)
        seq_len = tf.shape(embed)[1]
        return embed + pos_encoding[:, :seq_len, :]


    def position_wise_fc(self, vector):
        out = self.position_wise_fc_1(vector) # (batch, seq, dim_ffc)
        out = self.position_wise_fc_2(out) # (batch, seq, model_dim)
        return out


    def sub_layer(self, x, training=False, padding_mask=None):
        out_1, attention_weight = self.mha(x, K = x, V = x, mask=padding_mask, flag="encoder_mask")
        out_1 = self.dropout1(out_1, training=training)
        out_2 = self.layer_norm_1(out_1 + x)
        out_3 = self.position_wise_fc(out_2)
        out_3 = self.dropout2(out_3, training=training)
        out_4 = self.layer_norm_2(out_2 + out_3)

        return out_4, attention_weight


    def call(self, inputs, training=False, mask=None):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        # print("inputs: ", inputs)
        self.maxlen = tf.shape(inputs)[1]

        x = self.embed(inputs)  # (batch, seq, word_embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = self.add_positional_encoding(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


def get_embedding(inputs, token_num, pos_num, embed_dim, dropout_rate=0.1, trainable=True):
    """Get embedding layer.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param inputs: Input layers.
    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param embed_dim: The dimension of all embedding layers.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: The merged embedding layer and weights of token embedding.
    """

    embeddings = [
        TokenEmbedding(
            input_dim=token_num,
            output_dim=embed_dim,
            mask_zero=True,
            trainable=trainable,
            name='Embedding-Token',
        )(inputs[0]),
        keras.layers.Embedding(
            input_dim=2,
            output_dim=embed_dim,
            trainable=trainable,
            name='Embedding-Segment',
        )(inputs[1]),
    ]
    embeddings[0], embed_weights = embeddings[0]
    # embed_layer = PositionalEncoding(token_num, embed_dim)(embeddings[1])# inputs[0])
    # position = keras.layers.Lambda( lambda x: tf.range(0, x.shape[1], 1) )(inputs[0])
    # MIN MAX (MAX_POS_SIZE)
    """
    # TODO: DEBUG TF RANGE SHIT
    # TODO: DEBUG POSITION EMBEDDINGS
    # TODO: DEBUG POSITION EMBEDDINGS
    embed_layer = keras.layers.Embedding(
        input_dim=token_num,
        output_dim=embed_dim,
        trainable=trainable,
        name='Embedding-Position',
    )(position)
    """
    """
    embed_layer = PositionEmbedding(
        input_dim=200, # None, # pos_num,
        output_dim=embed_dim,
        mode=PositionEmbedding.MODE_ADD,
        trainable=trainable,
        name='Embedding-Position',
    )(inputs[0]) # TODO: MAKE THIS POSITIONAL!!!!
    """
    # TODO: DEBUG POSITION EMBEDDINGS
    # TODO: DEBUG POSITION EMBEDDINGS
    # TODO: DEBUG TF RANGE SHIT


    embeddings = [embeddings[0], embeddings[1]]
    embed_layer = keras.layers.Add(name='Embedding-Token-Segment')(embeddings)

    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='Embedding-Dropout',
        )(embed_layer)
    else:
        dropout_layer = embed_layer
    norm_layer = LayerNormalization(
        trainable=trainable,
        name='Embedding-Norm',
    )(dropout_layer)
    return norm_layer, embed_weights


class EmbeddingSimilarity(keras.layers.Layer):
    """Calculate similarity between features and token embeddings with bias term."""

    def __init__(self,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param output_dim: Same as embedding output dimension.
        :param initializer: Initializer for bias.
        :param regularizer: Regularizer for bias.
        :param constraint: Constraint for bias.
        :param kwargs: Arguments for parent class.
        """
        super(EmbeddingSimilarity, self).__init__(**kwargs)
        self.supports_masking = True
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self):
        config = {
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(int(input_shape[1][0]),),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='bias',
        )
        super(EmbeddingSimilarity, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (input_shape[1][0],)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        inputs, embeddings = inputs
        outputs = K.bias_add(K.dot(inputs, K.transpose(embeddings)), self.bias)
        return keras.activations.softmax(outputs)
