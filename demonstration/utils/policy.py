#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from utils.model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet),])


class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')
        self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

        con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.obj_v, self.con_v, self.policy,)
        self.optimizers = (self.obj_value_optimizer, self.con_value_optimizer, self.policy_optimizer)

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_obj_v(self, obs):
        with self.tf.name_scope('compute_obj_v') as scope:
            return tf.squeeze(self.obj_v(obs), axis=1)

    @tf.function
    def compute_con_v(self, obs):
        with self.tf.name_scope('compute_con_v') as scope:
            return tf.squeeze(self.con_v(obs), axis=1)

