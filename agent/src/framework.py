#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import client
import numpy as np


class Framework(object):

    def __init__(self, conf):
        self.turn_length = conf['turn_len']
        self.dialogue_length = conf['dialogue_len']
        self.actions = conf['actions']
        self.debug = conf['debug']

        self.dialogue_step = 0
        self.now_messages = []
        self.symbols = ['empty_mark']
        self.symbol_mapping = {'empty_mark': 0}

    def new_dialogue(self, file=None):
        """
        new dialogue(game)
        :return: state, reward, terminal
        """
        self.dialogue_step = 1
        message, reward, terminal = client.server_interact('new_dialogue', 'nothing', 'nothing', file=file)
        self.now_messages = [message]

        state = self.get_state()

        if self.debug:
            # print("send new dialogue action")
            print("receive message: ", message)
            print("reward: ", str(reward))
            print("terminal: ", str(terminal))
            # print("state: ", state)

        return state, reward, terminal

    def step_dialogue(self, ask_index, confirm_index, file=None):
        """
        move one step in now dialogue
        :param ask_index: ask index
        :param confirm_index: confirm index
        :param file: None if training else game log pointer
        :return: state, reward, terminal
        """
        self.dialogue_step += 1
        ask_index = int(ask_index)
        confirm_index = int(confirm_index)
        message, reward, terminal = client.server_interact('action', self.actions[ask_index],
                                                           self.actions[confirm_index], file=file)
        #if reward > -0.05:
        #    self.now_messages.append(message)
        self.now_messages.append(message)

        state = self.get_state()
        if self.dialogue_step >= self.dialogue_length:
            terminal = 1

        if self.debug:
            # print("send action: ask " + ask + " confirm " + confirm)
            print("receive message: ", message)
            print("reward: ", str(reward))
            print("terminal: ", str(terminal))
            # print("state: ", state)
            pass

        return state, reward, terminal

    def get_state(self):
        """
        construct now message(include history) state matrix
        :return: matrix, shape: self.dialogue_length * self.turn_length
        """
        # return self.parse_message(self.now_messages)
        return self.parse_message()

    def parse_message(self):
        """
        construct each turn message to tensor
        :return: np.array 1-D
        """
        state = np.zeros([self.dialogue_length, self.turn_length])

        for i in range(len(self.now_messages)):
            msg = self.now_messages[i]
            content = msg.split(' ')
            for j in range(len(content)):
                word = content[j]
                if word not in self.symbols:
                    self.symbols.append(word)
                    self.symbol_mapping[word] = self.symbols.index(word)
                state[i][j] = self.symbol_mapping[word]

        state.astype('int32')
        return state
