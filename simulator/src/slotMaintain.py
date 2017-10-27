#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import os
import sys
import random
import json
import xml.etree.cElementTree as ET

_ENCODE = 'utf-8'
_DECODE = 'utf-8'


# slot maintain
class SM(object):
    def __init__(self, args):
        self.slots = args['slots']
        self.template_path = args['setting_path']
        self.slot_num = len(args['slots'])

        self.slots_val = {}
        self.selected_true_value = []
        self.selected_alias_value = []
        self.change_mark = []
        self.not_provided = []
        self.provided_mark = []
        self.confirmed_mark = []
        self.userCmd = {"inform": "nothing",
                        "confirm": "nothing",
                        "inform_val": "nothing",
                        "confirm_val": "nothing",
                        "reject": False,
                        "hello": False}
        self.agentCmd = {"ask": "nothing",
                         "confirm": "nothing",
                         "confirm_val": "nothing",
                         "hello": False}
        self.random_seq = []

        self.__reward = 0
        self.__terminal = 0

        self.load_template()

    def clear_state(self):
        self.provided_mark = [0] * self.slot_num
        self.confirmed_mark = [0] * self.slot_num
        self.not_provided = [1] * self.slot_num
        self.__reward = -1
        self.__terminal = 0

    def slot_init(self):
        self.selected_true_value = []
        self.selected_alias_value = []
        self.change_mark = []
        for slot in self.slots:
            all_slots = self.slots_val[slot]
            # print(all_slots)
            index = random.randint(0, len(all_slots['true_value']) - 1)
            self.selected_true_value.append(all_slots['true_value'][index])
            self.selected_alias_value.append(all_slots['alias_value'][index])
            if random.uniform(0, 1) < 0.15:
                self.change_mark.append(1)
            else:
                self.change_mark.append(0)

    def cmd_init(self):
        self.userCmd = {"inform": "nothing",
                        "confirm": "nothing",
                        "inform_val": "nothing",
                        "confirm_val": "nothing",
                        "reject": False,
                        "hello": False}
        self.agentCmd = {"ask": "nothing",
                         "confirm": "nothing",
                         "confirm_val": "nothing",
                         "hello": False}

    def deal_action(self, action, ask, confirm):
        if action == "new_dialogue":
            self.new_dialogue()
        else:
            self.process_agent_action(ask, confirm)
        self.update_terminal()
        userCmd = self.userCmd
        agentCmd = self.agentCmd
        # self.track_state()
        return userCmd, agentCmd

    def update_terminal(self):
        self.__terminal = 1
        for mark in self.confirmed_mark:
            if mark != 1:
                self.__terminal = 0
        if self.__terminal and self.__reward > -0.05:
            self.__reward = 1

        # if self.reward < -0.05:
        #     self.__terminal = 1

    def new_dialogue(self):
        if random.uniform(0, 1) <= 1:
            self.clear_state()
            self.slot_init()
            self.cmd_init()
            self.userCmd['hello'] = True
            self.agentCmd['hello'] = True
            self.__reward = -0.01
        else:
            # random start
            pass

    def process_agent_action(self, ask, confirm):
        self.cmd_init()
        self.__reward = -0.01

        ask_index = -1 if ask == "nothing" else self.slots.index(ask)
        confirm_index = -1 if confirm == "nothing" else self.slots.index(confirm)
        # print(self.object)
        # print(ask, confirm)
        # print(ask_index, confirm_index)
        # print(self.provided_mark)

        if ask == "nothing" and confirm == "nothing":
            self.__reward = -1
        elif ask == confirm:
            self.__reward = -1
        elif ask != 'nothing' and self.provided_mark[ask_index] == 1:
            self.__reward = -1
        elif ask != 'nothing' and self.confirmed_mark[ask_index] == 1:
            self.__reward = -1
        elif confirm != 'nothing' and self.confirmed_mark[confirm_index] == 1:
            self.__reward = -1
        elif confirm != 'nothing' and self.not_provided[confirm_index] == 1:
            self.__reward = -1

        self.agentCmd['ask'] = ask
        self.agentCmd['confirm'] = confirm

        if confirm != 'nothing':
            if self.provided_mark[confirm_index] == 1 or self.confirmed_mark[confirm_index] == 1:
                self.agentCmd['confirm_val'] = self.selected_true_value[confirm_index]
                if self.change_mark[confirm_index] == 1:
                    self.agentCmd['confirm_val'] = self.selected_alias_value[confirm_index]

        # print(self.agentCmd['confirm_val'])

        if self.__reward > -0.1:
            # deal with user inform
            if ask != 'nothing':
                self.userCmd['inform'] = ask
                if self.change_mark[ask_index] == 1:
                    self.userCmd['inform_val'] = self.selected_alias_value[ask_index]
                else:
                    self.userCmd['inform_val'] = self.selected_true_value[ask_index]
                self.not_provided[ask_index] = 0
                self.provided_mark[ask_index] = 1
            # user confirm
            if confirm != 'nothing':
                if self.change_mark[confirm_index] == 1:
                    # 0.25 probability tell the true value
                    if random.uniform(0, 1) < 0.25:
                        # user should tell agent the true value
                        self.userCmd['inform'] = confirm
                        self.userCmd['inform_val'] = self.selected_true_value[confirm_index]
                    # 0.75 probability reject
                    else:
                        self.userCmd['reject'] = True
                        self.provided_mark[confirm_index] = 0
                        self.not_provided[confirm_index] = 1

                    self.change_mark[confirm_index] = 0
                else:
                    # user have to choice use general confirm or special confirm
                    self.userCmd['confirm'] = confirm
                    if random.uniform(0, 1) < 0.4:
                        self.userCmd['confirm_val'] = self.selected_true_value[confirm_index]
                    # change provided and confirmed mark
                    self.provided_mark[confirm_index] = 0
                    self.confirmed_mark[confirm_index] = 1

    @property
    def reward(self):
        return self.__reward

    @property
    def terminal(self):
        return self.__terminal

    def load_template(self):
        root = ET.parse(self.template_path)
        self.parse_slot(root)

    def parse_slot(self, slot_xml):
        for item in self.slots:
            true_value = []
            alias_values = []
            object_xml = slot_xml.find(item)
            templates = object_xml.findall('val')
            for template in templates:
                true_value.append(template.find('candidate').text)
                alias_values.append(template.find('alias').text)
            object_dict = {"true_value": true_value, "alias_value": alias_values}
            self.slots_val[item] = object_dict

    def track_state(self):
        print(self.slots)
        print(self.not_provided)
        print(self.provided_mark)
        print(self.confirmed_mark)


def main():
    args = {'slots': ["time", "number", "money", "duration", "location"], 'setting_path': "../setting/xml_slot.xml"}
    sm = SM(args)
    print(json.dumps(sm.slots, ensure_ascii=False).encode(_ENCODE))


if __name__ == '__main__':
    main()
