#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import agent


def main():
    conf = {'max_step': 1000000,
            'learn_start': 50000,
            'prioritized_learnt_start': 2500,
            'train_report_step': 200,
            'evaluate_step': 1000,
            'test_step': 500,
            'debug': False,
            'turn_len': 40,
            'dialogue_len': 10,
            'word_dim': 32,
            'keep_pro': 0.8,
            'mlp_hidden_unit': 64,
            'ep_start': 1,
            'ep_end': 0.1,
            'ep_step': 50000,
            'discount': 0.98,
            'update_freq': 4,
            'max_reward': 1,
            'min_reward': -0.1,
            'num_actions': 3,
            'actions': ['nothing', 'time', 'location'],
            'batch_size': 32,
            'target_q_clone_step': 1000,
            'replay_memory_size': 20000,
            'word_size': 500,
            'clip_delta': 1,
            'lr': 0.00008}

    machine_service = agent.Agent(conf)
    machine_service.training()


if __name__ == '__main__':
    main()
