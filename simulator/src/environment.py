#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:

import agentGenerator
import slotMaintain
import userGenerator

_ENCODE = 'utf-8'
_DECODE = 'utf-8'


class Environment(object):
    def __init__(self, slots):
        self.params = {'slots': slots,
                       'ag_path': '../setting/xml_agent.xml',
                       'ug_path': '../setting/xml_user.xml',
                       'sl_path': '../setting/xml_slot.xml'}
        self.ag = agentGenerator.AG({'slots': self.params['slots'],
                                     'setting_path': self.params['ag_path']})
        self.ug = userGenerator.UG({'slots': self.params['slots'],
                                    'setting_path': self.params['ug_path']})
        self.sm = slotMaintain.SM({'slots': self.params['slots'],
                                   'setting_path': self.params['sl_path']})
        self.actions = ['new_dialogue', 'nothing', 'action']
        self.local_state = {}

    def react(self, action, ask, confirm):
        if action is "getAgain":
            return self.local_state

        user_text = None
        agent_text = None
        reward = None
        terminal = None
        if action not in self.actions:
            message = 'error action'
            state = 'failed'
        elif ask != "nothing" and ask not in self.params['slots']:
            message = 'error ask'
            state = 'failed'
        elif confirm != "nothing" and confirm not in self.params['slots']:
            message = 'error confirm'
            state = 'failed'
        else:
            userCmd, agentCmd = self.sm.deal_action(action, ask, confirm)
            agent_text = self.ag.text_response(agentCmd['ask'], agentCmd['confirm'], agentCmd['confirm_val'], agentCmd['hello'])
            user_text = self.ug.text_response(userCmd['inform'], userCmd['confirm'], userCmd['inform_val'], userCmd['confirm_val'], userCmd['reject'], userCmd['hello'])
            reward = self.sm.reward
            terminal = self.sm.terminal
            state = 'success'
            message = 'success'
        
        self.local_state = {'state': state, 'message': message, 'agent_text': agent_text,
                            'user_text': user_text, 'reward': reward, 'terminal': terminal}

        return self.local_state

    def get_state(self):
        return self.local_state

    def print_state(self):
        print(' '.join(['agent: ', self.local_state['agent_text']]))
        print(' '.join(['user: ', self.local_state['user_text']]))
        print(' '.join(['reward: ', str(self.local_state['reward'])]))
        print(' '.join(['terminal: ', str(self.local_state['terminal'])]))


def main():
    pass


if __name__ == '__main__':
    main()
