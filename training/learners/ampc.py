#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import numpy as np
from gym.envs.user_defined.rlc.dynamics_and_models import EnvironmentModel
from training.preprocessor import Preprocessor
from training.utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.batch_data_lstm = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = EnvironmentModel(**args2envkwargs(args))
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data_lstm(self, batch_data, rb):
        # the size of batch_data is [6, batch_size, 29, dimensions]
        self.batch_data_lstm = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           'batch_ref_index': batch_data[5].astype(np.int32)
                           }

    def get_batch_data(self, batch_data, rb):
        # the size of batch_data is [6, batch_size, dimensions]
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           'batch_ref_index': batch_data[5].astype(np.int32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite//interval, self.tf.float32))
        return pf

    def model_rollout_for_update(self, start_obses, ite, mb_ref_index):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses, mb_ref_index)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        punish_terms_for_training_sum = self.tf.zeros((start_obses.shape[0],))
        real_punish_terms_sum = self.tf.zeros((start_obses.shape[0],))
        veh2veh4real_sum = self.tf.zeros((start_obses.shape[0],))
        veh2road4real_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses  # obses.shape = []
        pf = self.punish_factor_schedule(ite)
        processed_obses = self.preprocessor.tf_process_obses(obses)
        obj_v_pred = self.policy_with_value.compute_obj_v(processed_obses)
        # con_v_pred = self.policy_with_value.compute_con_v(processed_obses)

        # LSTMNet loss
        lstm_obs = self.tf.constant(self.batch_data_lstm['batch_obs'])  # [batch_size, 29, dimensions]
        surroundings_loss = []

        for i in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses, rewards, punish_terms_for_training, real_punish_term, veh2veh4real, veh2road4real = self.model.rollout_out(actions, self.policy_with_value)
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            punish_terms_for_training_sum += punish_terms_for_training
            real_punish_terms_sum += real_punish_term
            veh2veh4real_sum += veh2veh4real
            veh2road4real_sum += veh2road4real
            # lstm part
            pred = self.policy_with_value.surroundings(
                self.tf.convert_to_tensor(lstm_obs[:, i:i + 4, (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)]))  # 最后的维度
            loss_square = self.tf.square(pred - lstm_obs[:, i + 4, 0:10])
            surroundings_loss.append(loss_square)


        # obj v loss
        obj_v_loss = self.tf.reduce_mean(self.tf.square(obj_v_pred - self.tf.stop_gradient(rewards_sum)))
        # con_v_loss = self.tf.reduce_mean(self.tf.square(con_v_pred - self.tf.stop_gradient(real_punish_terms_sum)))

        # pg loss
        obj_loss = -self.tf.reduce_mean(rewards_sum)
        punish_term_for_training = self.tf.reduce_mean(punish_terms_for_training_sum)
        punish_loss = self.tf.stop_gradient(pf) * punish_term_for_training
        pg_loss = obj_loss + punish_loss

        real_punish_term = self.tf.reduce_mean(real_punish_terms_sum)
        veh2veh4real = self.tf.reduce_mean(veh2veh4real_sum)
        veh2road4real = self.tf.reduce_mean(veh2road4real_sum)

        # lstm loss
        lstm_loss = self.tf.reduce_mean(surroundings_loss)

        return obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, pf, lstm_loss

    @tf.function
    def forward_and_backward(self, mb_obs, ite, mb_ref_index):
        with self.tf.GradientTape(persistent=True) as tape:
            obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, pf, lstm_loss\
                = self.model_rollout_for_update(mb_obs, ite, mb_ref_index)

        with self.tf.name_scope('policy_gradient') as scope:
            pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
        with self.tf.name_scope('obj_v_gradient') as scope:
            obj_v_grad = tape.gradient(obj_v_loss, self.policy_with_value.obj_v.trainable_weights)
        # with self.tf.name_scope('con_v_gradient') as scope:
        #     con_v_grad = tape.gradient(con_v_loss, self.policy_with_value.con_v.trainable_weights)
        with self.tf.name_scope('lstm_gradient')as scope:
            lstm_grad = tape.gradient(lstm_loss, self.lstm.tranable_weights)

        return pg_grad, obj_v_grad, obj_v_loss, obj_loss, \
               punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, pf, lstm_grad

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
                                  self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, iteration):  # 还没改所有的compute_gradient的输入参数
        # the input of this function/the shape of samples is [6, batch_size, 29, dimensions]
        self.get_batch_data_lstm(samples, rb)
        original_samples = samples[:,:,4,:]  # the size of original_samples is [6, batch_size, dimensions]
        self.get_batch_data(original_samples, rb)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])  # the size of mb_bos is [batch_size, dimensions]
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        with self.grad_timer:
            pg_grad, obj_v_grad, obj_v_loss, obj_loss, \
            punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, pf, lstm_grad =\
                self.forward_and_backward(mb_obs, iteration, mb_ref_index, )

            pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)
            obj_v_grad, obj_v_grad_norm = self.tf.clip_by_global_norm(obj_v_grad, self.args.gradient_clip_norm)
            # con_v_grad, con_v_grad_norm = self.tf.clip_by_global_norm(con_v_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_term_for_training=punish_term_for_training.numpy(),
            real_punish_term=real_punish_term.numpy(),
            veh2veh4real=veh2veh4real.numpy(),
            veh2road4real=veh2road4real.numpy(),
            punish_loss=punish_loss.numpy(),
            pg_loss=pg_loss.numpy(),
            obj_v_loss=obj_v_loss.numpy(),
            # con_v_loss=con_v_loss.numpy(),
            punish_factor=pf.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            obj_v_grad_norm=obj_v_grad_norm.numpy(),
            # con_v_grad_norm=con_v_grad_norm.numpy()
        ))

        grads = obj_v_grad + pg_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass
