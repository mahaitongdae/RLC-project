#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: model.py
# =====================================

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention

from model_utils import veh_positional_encoding, EncoderLayer

import numpy as np

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)



class MLPNet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                            dtype=tf.float32)
        self.hidden = Sequential([Dense(num_hidden_units,
                                        activation=hidden_activation,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                        dtype=tf.float32) for _ in range(num_hidden_layers-1)])
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        self.outputs = Dense(output_dim,
                             activation=output_activation,
                             kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                             bias_initializer=tf.keras.initializers.Constant(0.),
                             dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.outputs(x)
        return x


class AttnNet(Model):
    def __init__(self, ego_dim, total_veh_dim, veh_num, tracking_dim,
                 num_attn_layers, d_model, d_ff, num_heads, dropout,
                 max_len=10, **kwargs):
        super(AttnNet, self).__init__(name=kwargs['name'])

        assert total_veh_dim / veh_num == 0
        self.ego_dim = ego_dim
        self.veh_num = veh_num
        self.veh_dim = total_veh_dim // veh_num
        self.tracking_dim = tracking_dim

        self.num_layers = num_attn_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.ego_embedding = Sequential([tf.keras.Input(shape=(self.ego_dim,)),
                                         Dense(units=d_model,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                               dtype=tf.float32)])
        self.vehs_embedding = Sequential([tf.keras.Input(shape=(self.veh_dim,)),
                                          Dense(units=d_model,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                                dtype=tf.float32)])

        self.pe = veh_positional_encoding(max_len, d_model)
        self.dropout = Dropout(self.dropout_rate)

        self.attn_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout)
                            for _ in range(self.num_layers-1)]
        self.out_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout)


    def call(self, x_ego, x_vehs, padding_mask, mu_mask, training=True):
        assert tf.shape(x_ego)[2] == self.ego_dim
        assert tf.shape(x_vehs)[2] == self.veh_dim
        assert tf.shape(x_vehs)[1] == self.veh_num

        seq_len = tf.shape(x_ego)[1] + tf.shape(x_vehs)[1]
        x1 = self.ego_embedding(x_ego)
        x2 = self.vehs_embedding(x_vehs)
        x = tf.concat([x1, x2], axis=1)
        assert tf.shape(x)[1] == seq_len
        x += self.pe[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers-1):
            x = self.attn_layers[i](x, training, padding_mask)

        output_mask = tf.logical_and(padding_mask, mu_mask)
        x, attn_weights = self.out_attn(x, x, attention_mask=output_mask,
                                        return_attention_scores=True, training=training)

        return x, attn_weights


def test_attrib():
    a = Variable(0, name='d')

    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(hasattr(p, 'get_weights'))
    print(hasattr(p, 'trainable_weights'))
    print(hasattr(a, 'get_weights'))
    print(hasattr(a, 'trainable_weights'))
    print(type(a))
    print(type(p))
    # print(a.name)
    # print(p.name)
    # p.build((None, 2))
    p.summary()
    # inp = np.random.random([10, 2])
    # out = p.forward(inp)
    # print(p.get_weights())
    # print(p.trainable_weights)


def test_clone():
    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(p._is_graph_network)
    s = tf.keras.models.clone_model(p)
    print(s)

def test_out():
    import numpy as np
    Qs = tuple(MLPNet(8, 2, 128, 1, name='Q' + str(i)) for i in range(2))
    inp = np.random.random((128, 8))
    out = [Q(inp) for Q in Qs]
    print(out)


def test_memory():
    import time
    Q = MLPNet(8, 2, 128, 1)
    time.sleep(111111)

def test_memory2():
    import time
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(30,), activation='relu'),
                                 tf.keras.layers.Dense(20, activation='relu'),
                                 tf.keras.layers.Dense(20, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu')])
    time.sleep(10000)

def test_attn():
    pass

if __name__ == '__main__':
    test_attn()
