#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:

import sys
import random
import numpy as np
import tensorflow as tf

import DQN
import replay_memory
from utility import list_to_tensor


class QLearner(object):

    def __init__(self, conf):
        self.conf = conf
        self.word_dim = conf['word_dim']
        self.word_size = conf['word_size']
        self.turn_len = conf['turn_len']
        self.dialogue_len = conf['dialogue_len']
        # epsilon setting
        self.ep_start = conf['ep_start']
        self.ep = conf['ep_start']
        self.ep_end = conf['ep_end']
        self.ep_step = conf['ep_step']

        self.discount = conf['discount']
        self.update_freq = conf['update_freq']
        self.max_reward = conf['max_reward']
        self.min_reward = conf['min_reward']
        self.num_actions = conf['num_actions']

        self.batch_size = conf['batch_size']
        self.learn_start = conf['learn_start']

        self.target_q_clone_step = conf['target_q_clone_step']

        self.debug = conf['debug']

        self.num_step = 0
        self.mini_batch_step = 0
        self.last_s = None
        self.last_ask = -1
        self.last_confirm = -1
        self.last_r = None
        self.last_t = None

        self.ask_loss = None
        self.confirm_loss = None
        self.v_ask_avg = 0
        self.v_confirm_avg = 0

        try:
            self.loss_log = open('../data/loss_log', 'w')
        except:
            print("open file failed!")
            sys.exit(1)

        replay_memory_conf = {'replay_memory_size': conf['replay_memory_size'],
                              'learn_start': conf['prioritized_learnt_start'],
                              'batch_size': conf['batch_size'],
                              'word_dim': conf['word_dim'],
                              'debug': conf['debug']}
        self.replay_memory = replay_memory.ReplayMemory(replay_memory_conf)

        embedding_init = np.random.rand(self.word_size, self.word_dim)
        embedding_init[0] *= 0
        embedding_init = embedding_init.astype('float32')

        output_network_conf = {'name': 'output_network',
                               'num_actions': conf['num_actions'],
                               'word_dim': conf['word_dim'],
                               'word_size': conf['word_size'],
                               'turn_len': conf['turn_len'],
                               'dialogue_len': conf['dialogue_len'],
                               'mlp_hidden_unit': conf['mlp_hidden_unit'],
                               'clip_delta': conf['clip_delta'],
                               'lr': conf['lr'],
                               'embedding_init': embedding_init}
        self.output_network = DQN.DQN(output_network_conf)
        target_network_conf = {'name': 'target_network',
                               'num_actions': conf['num_actions'],
                               'word_dim': conf['word_dim'],
                               'word_size': conf['word_size'],
                               'turn_len': conf['turn_len'],
                               'dialogue_len': conf['dialogue_len'],
                               'mlp_hidden_unit': conf['mlp_hidden_unit'],
                               'clip_delta': conf['clip_delta'],
                               'lr': conf['lr'],
                               'embedding_init': embedding_init}
        self.target_network = DQN.DQN(target_network_conf)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.writer = tf.train.SummaryWriter('../data/graph_logs', self.sess.graph)
        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)
        # self.sess.run(self.output_network.embedding_init)
        # self.sess.run(self.target_network.embedding_init)
        self.sync = self.sync_func()
        self.sess.run(self.sync)

    def q_update(self, reward, state, terminal):
        target_q_ask, target_q_confirm = self.sess.run(
            [self.target_network.ask_output, self.target_network.confirm_output],
            feed_dict={self.target_network.input_placeholder: state}
        )
        # Double Q find max actions
        output_q_ask, output_q_confirm = self.sess.run(
            [self.output_network.ask_output, self.output_network.confirm_output],
            feed_dict={self.output_network.input_placeholder: state}
        )
        output_max_ask_index, output_max_confirm_index = \
            self.find_max_action(output_q_ask, output_q_confirm, output_type='index')
        # Double Q get target Q val from output Q max index
        target_ask_max, target_confirm_max = \
            self.construct_output(output_max_ask_index, target_q_ask,
                                  output_max_confirm_index, target_q_confirm)

        # r + (1 - t) * discount * max_aQ(s2, a)
        target_ask = reward + target_ask_max * (1 - terminal)
        target_confirm = reward + target_confirm_max * (1 - terminal)
        return target_ask, target_confirm

    def convert_experience_to_separate(self, experience):
        last_s = list_to_tensor([item[0] for item in experience])
        reward = list_to_tensor([item[1] for item in experience])
        ask = list_to_tensor([item[2] for item in experience])
        confirm = list_to_tensor([item[3] for item in experience])
        state = list_to_tensor([item[4] for item in experience])
        terminal = list_to_tensor([item[5] for item in experience])
        return last_s, state, ask, confirm, reward, terminal

    def q_mini_batch(self):
        self.mini_batch_step += 1
        experience, w, rank_e_id = self.replay_memory.sample(self.num_step)
        # convert experience to s, s2, ask, confirm, reward, terminal
        s, s2, ask, confirm, r, t = self.convert_experience_to_separate(experience)

        if self.debug:
            print("s shape: " + str(s.shape))
            print("s2 shape: " + str(s2.shape))
            print("ask shape: " + str(ask.shape))
            print("confirm shape: " + str(confirm.shape))
            print("r shape: " + str(r.shape))
            print("t shape: " + str(t.shape))

        target_ask, target_confirm = self.q_update(r, s2, t)

        selected_ask_actions = self.construct_selected_action(ask)
        selected_confirm_actions = self.construct_selected_action(confirm)

        if self.debug:
            # print("Now do q_mini_batch")
            # print("s shape: " + str(s.shape))
            # print("target shape: " + str(target_ask.shape))
            # print("selected actions shape: " + str(selected_ask_actions.shape))
            # print("type of s: " + str(type(s)))
            # print("type of target: " + str(type(target_ask)))
            # print("type of selected_actions: " + str(type(selected_ask_actions)))
            pass

        feed_dict = {
            self.output_network.input_placeholder: s,
            self.output_network.ask_target_placeholder: target_ask,
            self.output_network.confirm_target_placeholder: target_confirm,
            self.output_network.selected_ask_placeholder: selected_ask_actions,
            self.output_network.selected_confirm_placeholder: selected_confirm_actions,
            self.output_network.w_placeholder: w
        }
        if self.mini_batch_step % 10 == 1:
            merged, update_delta, ask_loss, confirm_loss, _, _ = self.sess.run(
                [self.output_network.merged,
                 self.output_network.update_delta,
                 self.output_network.ask_loss_mean,
                 self.output_network.confirm_loss_mean,
                 self.output_network.run_ask_optimizer,
                 self.output_network.run_confirm_optimizer],
                feed_dict=feed_dict
            )
            self.writer.add_summary(merged, self.num_step)
        else:
            update_delta, ask_loss, confirm_loss, _, _ = self.sess.run(
                [self.output_network.update_delta,
                 self.output_network.ask_loss_mean,
                 self.output_network.confirm_loss_mean,
                 self.output_network.run_ask_optimizer,
                 self.output_network.run_confirm_optimizer],
                feed_dict=feed_dict
            )

        # update delta
        self.replay_memory.update_priority(rank_e_id, update_delta)

        self.ask_loss = ask_loss
        self.confirm_loss = confirm_loss
        self.loss_log.write('%f\t%d\n' % ((self.ask_loss + self.confirm_loss) / 2,
                                          self.mini_batch_step))

    def perceive(self, state, reward, terminal, testing=False, testing_ep=None):
        if self.max_reward:
            reward = min(reward, self.max_reward)
        if self.min_reward:
            reward = max(reward, self.min_reward)

        if self.last_s is not None and not testing:
            # construct replay
            experience = (self.last_s, reward, self.last_ask, self.last_confirm, state, terminal)
            self.replay_memory.store(experience)
            # debug
            if self.debug and terminal == 1:
                print('add terminal = 1 experience')

        # if self.num_step == self.learn_start + 1 and not testing:
        #     self.sample_validation_data()

        ask_index = -1
        confirm_index = -1
        if terminal != 1:
            ask_index, confirm_index = self.e_greedy(state, testing, testing_ep)

        if terminal != 1:
            self.last_s = np.copy(state)
            self.last_ask = ask_index
            self.last_confirm = confirm_index
        else:
            self.last_s = None
            self.last_ask = -1
            self.last_confirm = -1

        if self.num_step > self.learn_start and not testing and self.num_step % self.update_freq == 0:
            self.q_mini_batch()

        if not testing:
            self.num_step += 1

        if not testing and self.num_step > self.learn_start and self.num_step % self.target_q_clone_step == 0:
            # do the target network copy operation
            self.sess.run(self.sync)
            self.replay_memory.rebalance()
            print('sync output_network to target_network and rebalance replay memory!\n')

        if terminal != 1:
            return ask_index, confirm_index
        else:
            return -1, -1

    def e_greedy(self, state, testing, testing_ep):
        tmp_ep = self.ep_end + \
                 (self.ep_start - self.ep_end) * \
                 max(0, (self.ep_step - max(0, self.num_step - self.learn_start))) / self.ep_step
        self.ep = testing_ep or tmp_ep

        if random.random() < self.ep:
            # random action
            ask_index = random.randint(0, self.num_actions - 1)
            confirm_index = random.randint(0, self.num_actions - 1)
            while confirm_index == ask_index:
                confirm_index = random.randint(0, self.num_actions - 1)

            if self.debug:
                print('RANDOM ask index is: %d, confirm index is: %d' % (ask_index, confirm_index))
            return ask_index, confirm_index
        else:
            return self.greedy(state)

    def greedy(self, state):
        feet_dict = {
            self.output_network.input_placeholder: state.reshape([1, self.dialogue_len, self.turn_len])
        }

        target_q_ask, target_q_confirm = self.sess.run(
            [self.output_network.ask_output, self.output_network.confirm_output],
            feed_dict=feet_dict
        )

        ask_index, confirm_index = self.find_max_action(target_q_ask, target_q_confirm, 'index')

        if self.debug:
            print('GREEDY ask index is: %d, confirm index is: %d' % (ask_index[0], confirm_index[0]))

        return ask_index[0], confirm_index[0]

    @staticmethod
    def get_best_random(q_val_tensor):
        max_val = max(q_val_tensor)
        max_indexes = np.where(max_val == q_val_tensor)[0]
        max_index = max_indexes[random.randint(0, len(max_indexes) - 1)]
        return max_index, max_val

    def construct_selected_action(self, action_tensor):
        res = np.zeros([len(action_tensor), self.num_actions])
        for i in range(len(action_tensor)):
            res[i][int(action_tensor[i])] = 1

        return res.astype('float32')

    def find_max_action(self, ask, confirm, output_type='val'):
        batch_size = len(ask)
        ask_max_index = np.zeros([batch_size])
        ask_max_val = np.zeros([batch_size])
        confirm_max_index = np.zeros([batch_size])
        confirm_max_val = np.zeros([batch_size])
        for i in range(batch_size):
            tmp_ask = ask[i]
            tmp_confirm = confirm[i]
            ask_max_index[i], ask_max_val[i] = self.get_best_random(tmp_ask)

            confirm_min = min(tmp_confirm)
            tmp_confirm[int(ask_max_index[i])] = confirm_min
            confirm_max_index[i], confirm_max_val[i] = self.get_best_random(tmp_confirm)

        if self.debug:
            # print('ask q is:', ask)
            # print('confirm q is:', confirm)
            # print('max ask index:', ask_max_index)
            # print('max ask val:', ask_max_val)
            # print('max confirm index:', confirm_max_index)
            # print('max confirm val:', confirm_max_val)
            pass

        if output_type == 'val':
            return ask_max_val, confirm_max_val
        elif output_type == 'index':
            return ask_max_index, confirm_max_index

    @staticmethod
    def construct_output(ask_index, ask_val, confirm_index, confirm_val):
        batch_size = len(ask_index)
        ask = np.zeros([batch_size])
        confirm = np.zeros([batch_size])
        for i in range(batch_size):
            ask[i] = ask_val[i][int(ask_index[i])]
            confirm[i] = confirm_val[i][int(confirm_index[i])]
        return ask, confirm

    def zero_state(self):
        return np.zeros([self.word_dim])

    @staticmethod
    def sync_func():
        variables = ['embedding', 'turn_lstm/RNN/LSTMCell/W_0', 'turn_lstm/RNN/LSTMCell/B',
                     'dialogue_lstm/RNN/LSTMCell/W_0', 'dialogue_lstm/RNN/LSTMCell/B',
                     'ask_layer_1_weights', 'ask_layer_1_bias', 'ask_layer_2_weights',
                     'ask_layer_2_bias', 'confirm_layer_1_weights', 'confirm_layer_1_bias',
                     'confirm_layer_2_weights', 'confirm_layer_2_bias']
        sync = []
        for item in variables:
            with tf.variable_scope('output_network', reuse=True):
                output_weight = tf.get_variable(item)
            with tf.variable_scope('target_network', reuse=True):
                target_weight = tf.get_variable(item)
            sync.append(target_weight.assign(output_weight))

        return sync
