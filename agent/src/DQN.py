#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 


import functools
import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class DQN:

    def __init__(self, conf):
        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.word_dim = conf['word_dim']
        self.word_size = conf['word_size']

        self.mlp_hidden_unit = conf['mlp_hidden_unit']

        self.turn_len = conf['turn_len']
        self.dialogue_len = conf['dialogue_len']
        self.clip_delta = conf['clip_delta']
        # self.keep_pro = conf['keep_pro']

        self.lr = conf['lr']

        self.embedding_init = conf['embedding_init']

        with tf.variable_scope(self.name):
            # placeholder
            self.input_placeholder = tf.placeholder(tf.int32,
                                                    [None, self.dialogue_len, self.turn_len])
            self.selected_ask_placeholder = tf.placeholder(tf.float32,
                                                           [None, self.num_actions],
                                                           name='selected_inform')
            self.selected_confirm_placeholder = tf.placeholder(tf.float32,
                                                               [None, self.num_actions],
                                                               name='selected_confirm')
            self.w_placeholder = tf.placeholder(tf.float32, [None])
            self.ask_target_placeholder = tf.placeholder(tf.float32, [None])
            self.confirm_target_placeholder = tf.placeholder(tf.float32, [None])

            # embedding
            self.embedding = tf.get_variable('embedding',
                                             initializer=tf.constant(self.embedding_init))
            # get input data
            self.input_data = tf.nn.embedding_lookup(self.embedding, self.input_placeholder)

            self.turn_hidden_B_T_D = self.turn_lstm(self.input_data)

            self.last_dialogue_output = self.dialogue_lstm(self.turn_hidden_B_T_D)

            # ask
            layer_name = 'ask_layer_1'
            hidden = self.mlp_hidden_unit
            self.ask_w_1 = tf.get_variable(
                '' + layer_name + '_weights',
                [self.word_dim, hidden],
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            self.ask_b_1 = tf.get_variable(
                '' + layer_name + '_bias',
                [hidden],
                initializer=tf.constant_initializer(0.1)
            )
            self.ask_hidden_layer = tf.add(
                tf.matmul(self.last_dialogue_output, self.ask_w_1),
                self.ask_b_1
            )
            self.ask_hidden_output = tf.nn.relu(self.ask_hidden_layer)

            layer_name = 'ask_layer_2'
            dim = self.mlp_hidden_unit
            hidden = self.num_actions
            self.ask_w_2 = tf.get_variable(
                '' + layer_name + '_weights',
                [dim, hidden],
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            self.ask_b_2 = tf.get_variable(
                '' + layer_name + '_bias',
                [hidden],
                initializer=tf.constant_initializer(0.1)
            )
            self.ask_output = tf.add(
                tf.matmul(self.ask_hidden_output, self.ask_w_2),
                self.ask_b_2
            )

            # confirm
            layer_name = 'confirm_layer_1'
            hidden = self.mlp_hidden_unit
            self.confirm_w_1 = tf.get_variable(
                '' + layer_name + '_weights',
                [self.word_dim, hidden],
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            self.confirm_b_1 = tf.get_variable(
                '' + layer_name + '_bias',
                [hidden],
                initializer=tf.constant_initializer(0.1)
            )
            self.confirm_hidden_layer = tf.add(
                tf.matmul(self.last_dialogue_output, self.confirm_w_1),
                self.confirm_b_1
            )
            self.confirm_hidden_output = tf.nn.relu(self.confirm_hidden_layer)

            layer_name = 'confirm_layer_2'
            dim = self.mlp_hidden_unit
            hidden = self.num_actions
            self.confirm_w_2 = tf.get_variable(
                '' + layer_name + '_weights',
                [dim, hidden],
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            self.confirm_b_2 = tf.get_variable(
                '' + layer_name + '_bias',
                [hidden],
                initializer=tf.constant_initializer(0.1)
            )
            self.confirm_output = tf.add(
                tf.matmul(self.confirm_hidden_output, self.confirm_w_2),
                self.confirm_b_2
            )

            # ask selected action
            ask_selected_action = tf.reduce_sum(
                tf.mul(self.ask_output, self.selected_ask_placeholder),
                reduction_indices=1
            )

            ask_diff = tf.sub(self.ask_target_placeholder, ask_selected_action)

            if self.clip_delta > 0:
                temp = tf.minimum(tf.abs(ask_diff), tf.constant(self.clip_delta, dtype=tf.float32))
                self.ask_loss = 0.5 * tf.pow(temp, 2) * self.w_placeholder
            else:
                self.ask_loss = 0.5 * tf.pow(ask_diff, 2) * self.w_placeholder

            self.ask_loss_mean = tf.reduce_mean(self.ask_loss)
            self.ask_loss_summary = tf.scalar_summary('ask_loss', self.ask_loss_mean)

            # confirm selected action
            confirm_selected_action = tf.reduce_sum(
                tf.mul(self.confirm_output, self.selected_confirm_placeholder),
                reduction_indices=1
            )

            confirm_diff = tf.sub(self.confirm_target_placeholder, confirm_selected_action)

            if self.clip_delta > 0:
                temp = tf.minimum(tf.abs(confirm_diff),
                                  tf.constant(self.clip_delta, dtype=tf.float32))
                self.confirm_loss = 0.5 * tf.pow(temp, 2) * self.w_placeholder
            else:
                self.confirm_loss = 0.5 * tf.pow(confirm_diff, 2) * self.w_placeholder

            self.confirm_loss_mean = tf.reduce_mean(self.confirm_loss)
            self.confirm_loss_summary = tf.scalar_summary('confirm_loss', self.confirm_loss_mean)

            # get delta update
            self.update_delta = tf.add(tf.abs(ask_diff), tf.abs(confirm_diff)) / 2

            # optimizer
            self.ask_optimizer = tf.train.AdamOptimizer(self.lr)
            self.confirm_optimizer = tf.train.AdamOptimizer(self.lr)
            self.run_ask_optimizer = self.ask_optimizer.minimize(self.ask_loss)
            self.run_confirm_optimizer = self.confirm_optimizer.minimize(self.confirm_loss)
            self.run_optimizer = [self.run_ask_optimizer, self.run_confirm_optimizer]

            self.merged = tf.merge_all_summaries()

    def turn_lstm(self, data):
        # turn lstm
        turn_lstm_name = 'turn_lstm'
        data = tf.reshape(data, [-1, self.turn_len, self.word_dim])
        with tf.variable_scope(turn_lstm_name):
            turn_cell = tf.nn.rnn_cell.LSTMCell(self.word_dim,
                                                forget_bias=0.0,
                                                state_is_tuple=True)
            output, state = tf.nn.dynamic_rnn(turn_cell,
                                              data,
                                              sequence_length=self.length(data),
                                              dtype=tf.float32)

            last_turn_hidden = state[0]
            turn_hidden_B_T_D = tf.reshape(last_turn_hidden, [-1, self.dialogue_len, self.word_dim])

            return turn_hidden_B_T_D

    def dialogue_lstm(self, data):
        # dialogue lstm
        dialogue_lstm_name = 'dialogue_lstm'
        with tf.variable_scope(dialogue_lstm_name):
            dialogue_cell = tf.nn.rnn_cell.LSTMCell(self.word_dim,
                                                    forget_bias=0.0,
                                                    state_is_tuple=True)
            output, state = tf.nn.dynamic_rnn(dialogue_cell,
                                              data,
                                              sequence_length=self.length(data),
                                              dtype=tf.float32)
            last_dialogue_output = state[1]

        return last_dialogue_output

    def length(self, data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
