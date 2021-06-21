#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: buffer.py
# =====================================

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, args, buffer_id):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self.args = args
        self.buffer_id = buffer_id
        self._storage = []
        self._maxsize = self.args.max_buffer_size
        self._next_idx = 0
        self.replay_starts = self.args.replay_starts
        self.replay_batch_size = self.args.replay_batch_size
        self.stats = {}
        self.replay_times = 0
        logger.info('Buffer initialized')

    def get_stats(self):
        self.stats.update(dict(storage=len(self._storage)))
        return self.stats

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, ref_index, weight):
        data = (obs_t, action, reward, obs_tp1, done, ref_index)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, ref_indexs = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, ref_index = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            ref_indexs.append(ref_index)
        return np.array(obses_t), np.array(actions), np.array(rewards), \
               np.array(obses_tp1), np.array(dones), np.array(ref_indexs)

    def sample_idxes(self, batch_size):
        return np.array([random.randint(0, len(self._storage) - 1) for _ in range(batch_size)], dtype=np.int32)

    def sample_with_idxes(self, idxes):
        return list(self._encode_sample(idxes)) + [idxes,]

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_idxes(idxes)

    def add_batch(self, batch):
        for trans in batch:
            self.add(*trans, 0)

    def replay(self):
        if len(self._storage) < self.replay_starts:
            return None
        if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
            logger.info('Buffer info: {}'.format(self.get_stats()))

        self.replay_times += 1
        return self.sample(self.replay_batch_size) # here the size of return is [batch_size, 29, dimensions]

# class ReplayBufferPro(ReplayBuffer):
#     # this class will samples data while keeping 4 continuity
#         def sample_idxes(self, batch_size):
#             # make sure to keep continuity in future 4 steps
#             idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#             # firstly keep 4 continuity
#             for ith, idx in enumerate(idxes):
#                 temp = self.judge(idx)
#                 if temp != idx:
#                     idxes[ith] = temp
#             return np.array(idxes, dtype=np.int32)
#
#         def judge(self, idx):
#             if idx % 200 > 195:
#                 new_idx = random.randint(0, len(self._storage) - 1)
#                 self.judge(new_idx)
#             else:
#                 return idx
#
#         def sample(self, batch_size):
#             idxes = self.sample_idxes(batch_size)
#             return self.sample_with_idxes(idxes)
#
#         def replay(self):
#             if len(self._storage) < self.replay_starts:
#                 return None
#             if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
#                 logger.info('Buffer info: {}'.format(self.get_stats()))
#
#             self.replay_times += 1
#             return self.sample(self.replay_batch_size)
#
#         def replay_pro(self, samples):
#             indexs = samples[5]
#             obses_t, actions, rewards, obses_tp1, dones, ref_indexs = [], [], [], [], [], []
#             for idx in indexs:
#                 for i in range(4):
#                     # index: [1:4], obses = [s0:s3 + s4], where s4 is the predictive real value
#                     # s0, s1, s2, s3 are 4 historical values, where s3 is the current value
#                     data = self._storage[idx + i]
#                     obs_t, action, reward, obs_tp1, done, ref_index = data
#                     obses_t.append(np.array(obs_t, copy=False))
#                     actions.append(np.array(action, copy=False))
#                     rewards.append(reward)
#                     obses_tp1.append(np.array(obs_tp1, copy=False))
#                     dones.append(done)
#                     ref_indexs.append(ref_index)
#             return np.array(obses_t), np.array(actions), np.array(rewards), \
#                    np.array(obses_tp1), np.array(dones), np.array(ref_indexs)


class DistendReplyBuffer(ReplayBuffer):
    # this class distend the size of sampling
    # sample the past 4 moments + current moment + the future 24 moments

    def sample_idxes(self, batch_size):
        # make sure to select the idxes which have 29 continuous steps s
        idxes = [random.randint(4, len(self._storage) - 1) for _ in range(batch_size)]
        for ith, idx in enumerate(idxes):
            judgement = self.judge(idx)
            while judgement == False:
                new_index = random.randint(4, len(self._storage) - 1)
                judgement = self.judge(new_index)
            if new_index != idx:
                idxes[ith] = new_index
        return np.array(idxes, dtype=np.int32)

    def judge(self, idx):
        continuity_judge = True
        # current_ref = self._storage[idx][5]
        for i in range(28):
            if self._storage[idx-4+i][4] == True:
                continuity_judge = False
            return continuity_judge

    def sample_with_idxes(self, idxes):
        return list(self._encode_sample(idxes))

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_idxes(idxes)

    def _encode_sample(self, idxes):
        # 目标维度：[batch, 29，obs_dim]
        obses_t, actions, rewards, obses_tp1, dones, ref_indexs = [], [], [], [], [], []
        for idx in idxes:
            s_temp, a_temp, r_temp, ss_temp, ds_temp, ris_temp = [], [], [], [], [], []
            for i in range(29):
                data = self._storage[idx-4+i]
                obs_t, action, reward, obs_tp1, done, ref_index = data
                s_temp.append(np.array(obs_t, copy=False))
                a_temp.append(np.array(action, copy=False))
                r_temp.append(reward)
                ss_temp.append(np.array(obs_tp1, copy=False))
                ds_temp.append(done)
                ris_temp.append(ref_index)
            obses_t.append(np.array(s_temp, copy=False))
            actions.append(np.array(a_temp, copy=False))
            rewards.append(r_temp)
            obses_tp1.append(np.array(ss_temp, copy=False))
            dones.append(ds_temp)
            ref_indexs.append(ris_temp)
        return np.array(obses_t), np.array(actions), np.array(rewards), \
               np.array(obses_tp1), np.array(dones), np.array(ref_indexs)
