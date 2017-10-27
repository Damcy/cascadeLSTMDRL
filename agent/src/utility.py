#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:

import numpy as np


action_object = ['nothing', 'time', 'location']


def num_to_action(action_index):
    """
    find ask and confirm object according action_index
    :param action_index: int
    :return: list include ask object and confirm object
    """
    global action_object
    ask_index = action_index / 3
    confirm_index = action_index % 3
    return action_object[int(ask_index)], action_object[int(confirm_index)]


def action_to_num(ask_action, confirm_action):
    """
    convert ask and confirm objects to action_index
    :param ask_action: string
    :param confirm_action: string
    :return: int
    """
    global action_object

    if ask_action not in action_object:
        print("ask_action " + ask_action + "is not in action objects")
        return -1
    if confirm_action not in action_object:
        print("confirm_action " + ask_action + "is not in action objects")
        return -1

    ask_index = action_object.index(ask_action)
    confirm_index = action_object.index(confirm_action)

    return ask_index * 3 + confirm_index


def list_to_tensor(input_list):
    res = np.array(input_list)
    return res


def list_to_dict(in_list):
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict):
    return dict((in_dict[i], i) for i in in_dict)
