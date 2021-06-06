#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/6/3
# @Author  : Dongjie Yu (Tsinghua Univ.)
# @FileName: model_utils.py
# =====================================

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization
# from official.nlp.modeling.layers.position_embedding import RelativePositionEmbedding

import numpy as np
import math

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from utils.shapechecker import ShapeChecker


def get_angles(pos, i, d_model, max_len=1000):
    angle_rates = 1 / np.power(max_len, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def veh_positional_encoding(position, d_model):
    p = np.ones((position, 1), dtype=np.int8)
    p[0] = 0
    angle_rads = get_angles(p,
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


'''
class VehsPE(RelativePositionEmbedding):
    def __init__(self,
                 hidden_size,
                 min_timescale=1.0,
                 max_timescale=1.0e3,
                 **kwargs):
        super(VehsPE, self).__init__()
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale
        position = np.ones((int(max_timescale),), dtype=np.int8)
        position[0] = 0
        position = tf.convert_to_tensor(position, dtype=tf.float32)

        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) *
            -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        self.pe = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

    def call(self, inputs, length=None):
        assert inputs.shape[2] == self._hidden_size
        inputs = inputs + self.pe[:, :inputs.shape(1)]
        return inputs
'''


def pointwise_feedforward(d_model, d_ff):
    return Sequential([
        Dense(d_ff, activation='relu'),
        Dense(d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.ffn = pointwise_feedforward(d_model, d_ff)

        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2