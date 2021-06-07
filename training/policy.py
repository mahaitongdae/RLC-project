#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

from model import MLPNet
from model import AttnNet

NAME2MODELCLS = dict([('MLP', MLPNet), ('Attn', AttnNet)])


class AttnPolicy4Lagrange(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args

        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        mu_dim = self.args.con_dim
        veh_dim = self.args.veh_dim
        veh_num = self.args.veh_num
        ego_dim = self.args.ego_dim
        tracking_dim = self.args.tracking_dim

        d_model = self.args.d_model
        num_attn_layers = self.args.num_attn_layers
        d_ff = self.args.d_ff
        num_heads = self.args.num_heads
        dropout = self.args.drop_rate
        max_len = self.args.max_veh_num

        assert tracking_dim + ego_dim + veh_dim*veh_num == obs_dim
        assert 4 + veh_num * 4 == mu_dim

        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        backbone_cls = NAME2MODELCLS[self.args.backbone_cls]

        # Attention backbone
        self.backbone = backbone_cls(ego_dim, obs_dim-tracking_dim-ego_dim, veh_num, tracking_dim,
                                     num_attn_layers, d_model, d_ff, num_heads, dropout,
                                     max_len, name='backbone')
        mu_value_lr_schedule = PolynomialDecay(*self.args.mu_lr_schedule)
        self.mu_optimizer = self.tf.optimizers.Adam(mu_value_lr_schedule, name='mu_adam_opt')

        # self.policy = Sequential([tf.keras.layers.InputLayer(input_shape=(d_model,)),
        #                           Dense(d_model, activation=self.args.policy_out_activation,
        #                                 kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
        #                                 dtype=tf.float32),
        #                           Dense(act_dim * 2, activation=self.args.policy_out_activation,
        #                                 kernel_initializer=tf.keras.initializers.Orthogonal(1.),
        #                                 bias_initializer = tf.keras.initializers.Constant(0.),
        #                                 dtype = tf.float32),])
        self.policy = policy_model_cls(d_model, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        # self.value = Sequential([tf.keras.Input(shape=(d_model,)),
        #                          Dense(1, activation='linear',
        #                                kernel_initializer=tf.keras.initializers.Orthogonal(1.),
        #                                bias_initializer=tf.keras.initializers.Constant(0.),
        #                                dtype=tf.float32),])
        # value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.value_optimizer = self.tf.keras.optimizers.Adam(value_lr_schedule, name='v_adam_opt')

        self.models = (self.backbone, self.policy)
        self.optimizers = (self.mu_optimizer, self.policy_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        policy_len = len(self.policy.trainable_weights)
        policy_grad, mu_grad = grads[:policy_len], grads[policy_len:]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        if iteration % self.args.mu_update_interval == 0:
            self.mu_optimizer.apply_gradients(zip(mu_grad, self.backbone.trainable_weights))

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
    def compute_mu(self, obs, nonpadding_ind, training=True):
        def create_padding_mask(batch_size, seq_len, nonpadding_ind):
            nonpadding_ind = tf.cast(nonpadding_ind, dtype=tf.float32)
            nonpadding_ind = tf.concat([tf.ones((batch_size,1)), nonpadding_ind], axis=-1)
            nonpadding_ind = tf.reshape(nonpadding_ind, (batch_size, 1, -1))
            repaet_times = tf.constant([1, seq_len, 1], tf.int32)

            return tf.tile(nonpadding_ind, repaet_times)

        def create_mu_mask(batch_size, seq_len):
            mask = np.identity(seq_len, dtype=np.float32)
            mask[:, 0] = 1
            mask[0, :] = 1
            mask = mask[np.newaxis, :, :]
            return tf.convert_to_tensor(np.repeat(mask, repeats=batch_size, axis=0), dtype=tf.float32)

        with self.tf.name_scope('compute_mu') as scope:
            batch_size = (obs).shape[0]
            seq_len = self.args.veh_num+1
            x_ego = tf.expand_dims(obs[:, :self.args.ego_dim+self.args.tracking_dim], axis=1)
            x_vehs = tf.reshape(obs[:, self.args.ego_dim+self.args.tracking_dim:], (batch_size, -1, self.args.veh_dim))

            assert x_vehs.shape[1] == self.args.veh_num

            # hidden, attn_weights = self.backbone(x_ego, x_vehs,
            #                                      padding_mask=create_padding_mask(batch_size, seq_len, nonpadding_ind),
            #                                      mu_mask=create_mu_mask(batch_size, seq_len),
            #                                      training=training)
            hidden, attn_weights = self.backbone([x_ego, x_vehs,
                                                   create_padding_mask(batch_size, seq_len, nonpadding_ind),
                                                   create_mu_mask(batch_size, seq_len),],
                                                 training=training)
            mu_attn = attn_weights[:, :, 0, 1:]
            return hidden[:, 0, :], tf.cast(tf.exp(5*mu_attn)-1, dtype=tf.float32)

    @tf.function
    def compute_action(self, obs, nonpadding_ind, training=True):
        hidden, _ = self.compute_mu(obs, nonpadding_ind, training)
        hidden = tf.stop_gradient(hidden)
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(hidden)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    # @tf.function
    # def compute_v(self, hidden):
    #     with self.tf.name_scope('compute_v') as scope:
    #         return tf.squeeze(self.value(hidden), axis=1)


class Policy4Lagrange(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        mu_dim = self.args.con_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        mu_model_cls = NAME2MODELCLS[self.args.mu_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')
        self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        self.mu = mu_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, mu_dim, name='mu', output_activation=
                               self.args.mu_out_activation)

        mu_value_lr_schedule = PolynomialDecay(*self.args.mu_lr_schedule)
        self.mu_optimizer = self.tf.optimizers.Adam(mu_value_lr_schedule, name='mu_adam_opt')

        # obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')
        #
        # con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.policy, self.mu)
        self.optimizers = (self.policy_optimizer, self.mu) # self.obj_value_optimizer, self.con_value_optimizer,

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        policy_len = len(self.policy.trainable_weights)
        policy_grad, mu_grad = grads[:policy_len], grads[policy_len:]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        if iteration % self.args.mu_update_interval == 0:
            self.mu_optimizer.apply_gradients(zip(mu_grad, self.mu.trainable_weights))

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

    # @tf.function
    # def compute_obj_v(self, obs):
    #     with self.tf.name_scope('compute_obj_v') as scope:
    #         return tf.squeeze(self.obj_v(obs), axis=1)
    #
    # @tf.function
    # def compute_con_v(self, obs):
    #     with self.tf.name_scope('compute_con_v') as scope:
    #         return tf.squeeze(self.con_v(obs), axis=1)

    @tf.function
    def compute_mu(self, obs):
        with self.tf.name_scope('compute_mu') as scope:
            return self.mu(obs)

class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

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
        # self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

        # con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.obj_v, self.policy,)
        self.optimizers = (self.obj_value_optimizer, self.policy_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        obj_v_len = len(self.obj_v.trainable_weights)
        # con_v_len = len(self.con_v.trainable_weights)
        obj_v_grad, policy_grad = grads[:obj_v_len], grads[obj_v_len:]
        self.obj_value_optimizer.apply_gradients(zip(obj_v_grad, self.obj_v.trainable_weights))
        # self.con_value_optimizer.apply_gradients(zip(con_v_grad, self.con_v.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

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

    # @tf.function
    # def compute_con_v(self, obs):
    #     with self.tf.name_scope('compute_con_v') as scope:
    #         return tf.squeeze(self.con_v(obs), axis=1)

'''
def test_policy():
    import gym
    from train_script import built_mixedpg_parser
    args = built_mixedpg_parser()
    print(args.obs_dim, args.act_dim)
    env = gym.make('PathTracking-v0')
    policy = PolicyWithQs(env.observation_space, env.action_space, args)
    obs = np.random.random((128, 6))
    act = np.random.random((128, 2))
    Qs = policy.compute_Qs(obs, act)
    print(Qs)

def test_policy2():
    from train_script import built_mixedpg_parser
    import gym
    args = built_mixedpg_parser()
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)

def test_policy_with_Qs():
    from train_script import built_mixedpg_parser
    import gym
    import numpy as np
    import tensorflow as tf
    args = built_mixedpg_parser()
    args.obs_dim = 3
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)
    # print(policy_with_value.policy.trainable_weights)
    # print(policy_with_value.Qs[0].trainable_weights)
    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)

    with tf.GradientTape() as tape:
        acts, _ = policy_with_value.compute_action(obses)
        Qs = policy_with_value.compute_Qs(obses, acts)[0]
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy_with_value.policy.trainable_weights)
    print(gradient)

def test_mlp():
    import tensorflow as tf
    import numpy as np
    policy = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    value = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(4,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    print(policy.trainable_variables)
    print(value.trainable_variables)
    with tf.GradientTape() as tape:
        obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
        obses = tf.convert_to_tensor(obses)
        acts = policy(obses)
        a = tf.reduce_mean(acts)
        print(acts)
        Qs = value(tf.concat([obses, acts], axis=-1))
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy.trainable_weights)
    print(gradient)
'''
def test_attn_policy():
    import tensorflow as tf
    import numpy as np
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    def built_AMPC_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--mode', type=str, default='training')  # training testing

        # trainer
        parser.add_argument('--policy_type', type=str, default='Policy4Toyota')
        parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
        parser.add_argument('--evaluator_type', type=str, default='Evaluator')
        parser.add_argument('--buffer_type', type=str, default='normal')
        parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
        parser.add_argument('--off_policy', type=str, default=True)

        # env
        parser.add_argument('--env_id', default='CrossroadEnd2end-v20')
        parser.add_argument('--env_kwargs_num_future_data', type=int, default=0)
        parser.add_argument('--env_kwargs_training_task', type=str, default='left')
        parser.add_argument('--obs_dim', default=None)
        parser.add_argument('--act_dim', default=None)

        # learner
        parser.add_argument('--alg_name', default='AMPC')
        parser.add_argument('--M', type=int, default=1)
        parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[25])
        parser.add_argument('--gamma', type=float, default=1.)
        parser.add_argument('--gradient_clip_norm', type=float, default=10)
        parser.add_argument('--init_punish_factor', type=float, default=10.)
        parser.add_argument('--pf_enlarge_interval', type=int, default=20000)
        parser.add_argument('--pf_amplifier', type=float, default=1.)

        # worker
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--worker_log_interval', type=int, default=5)
        parser.add_argument('--explore_sigma', type=float, default=None)

        # buffer
        parser.add_argument('--max_buffer_size', type=int, default=50000)
        parser.add_argument('--replay_starts', type=int, default=3000)
        parser.add_argument('--replay_batch_size', type=int, default=256)
        parser.add_argument('--replay_alpha', type=float, default=0.6)
        parser.add_argument('--replay_beta', type=float, default=0.4)
        parser.add_argument('--buffer_log_interval', type=int, default=40000)

        # tester and evaluator
        parser.add_argument('--num_eval_episode', type=int, default=2)
        parser.add_argument('--eval_log_interval', type=int, default=1)
        parser.add_argument('--fixed_steps', type=int, default=50)
        parser.add_argument('--eval_render', type=bool, default=True)

        # policy and model
        parser.add_argument('--value_model_cls', type=str, default='MLP')
        parser.add_argument('--policy_model_cls', type=str, default='MLP')
        parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 150000, 1e-5])
        parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 150000, 1e-5])
        parser.add_argument('--num_hidden_layers', type=int, default=2)
        parser.add_argument('--num_hidden_units', type=int, default=256)
        parser.add_argument('--hidden_activation', type=str, default='elu')
        parser.add_argument('--deterministic_policy', default=True, action='store_true')
        parser.add_argument('--policy_out_activation', type=str, default='tanh')
        parser.add_argument('--action_range', type=float, default=None)

        # preprocessor
        parser.add_argument('--obs_preprocess_type', type=str, default='scale')
        parser.add_argument('--obs_scale', type=list, default=None)
        parser.add_argument('--reward_preprocess_type', type=str, default='scale')
        parser.add_argument('--reward_scale', type=float, default=1.)
        parser.add_argument('--reward_shift', type=float, default=0.)

        # optimizer (PABAL)
        parser.add_argument('--max_sampled_steps', type=int, default=0)
        parser.add_argument('--max_iter', type=int, default=150000)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_learners', type=int, default=30)
        parser.add_argument('--num_buffers', type=int, default=4)
        parser.add_argument('--max_weight_sync_delay', type=int, default=300)
        parser.add_argument('--grads_queue_size', type=int, default=20)
        parser.add_argument('--grads_max_reuse', type=int, default=20)
        parser.add_argument('--eval_interval', type=int, default=5000)
        parser.add_argument('--save_interval', type=int, default=5000)
        parser.add_argument('--log_interval', type=int, default=100)

        # Attention, added by YDJ
        parser.add_argument('--num_attn_layers', type=int, default=3)
        parser.add_argument('--con_dim', type=int, default=32)
        parser.add_argument('--d_model', type=int, default=128)
        parser.add_argument('--d_ff', type=int, default=256)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--drop_rate', type=float, default=0.1)
        parser.add_argument('--max_veh_num', type=int, default=10)
        parser.add_argument('--backbone_cls', type=str, default='Attn')
        parser.add_argument('--mu_lr_schedule', type=list, default=[8e-5, 150000, 1e-5])

        return parser.parse_args()

    args = built_AMPC_parser()

    args.veh_dim = 4 # env.per_veh_info_dim
    args.veh_num = 7 # env.veh_num
    args.ego_dim = 6 # env.ego_info_dim
    args.tracking_dim = 3 # env.per_tracking_info_dim

    args.obs_dim = args.ego_dim + args.tracking_dim + args.veh_num *args.veh_dim # env.per_veh_info_dim
    args.act_dim = 2
    policy_with_mu = AttnPolicy4Lagrange(args)

    g1 = tf.random.Generator.from_seed(1)
    obs = g1.normal(shape=[3, 37])
    nonpadding = tf.constant([[1,0,1,1,1,0,1], [1,0,0,0,1,1,1], [1,1,0,0,1,0,1]])

    print(policy_with_mu.compute_action(obs, nonpadding))

if __name__ == '__main__':
    test_attn_policy()
    # test_policy_with_Qs()
