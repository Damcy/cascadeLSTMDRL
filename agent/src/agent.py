#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description:

import time
import json
from collections import defaultdict
from tqdm import tqdm

import framework
import QLearner


class Agent(object):

    def __init__(self, conf):
        self.step = 0
        self.max_step = conf['max_step']
        self.learn_start = conf['learn_start']
        self.train_report_step = conf['train_report_step']
        self.evaluate_step = conf['evaluate_step']
        self.test_step = conf['test_step']

        self.framework = framework.Framework(conf)
        self.QLearner = QLearner.QLearner(conf)
        self.max_avg_reward = -1000
        self.best_network = None

        self.training_time = 0
        self.testing_time = 0
        self.run_test_cnt = 0

        self.reward_log = open('../data/reward_log', 'w')
        self.test_log = open('../data/game_log', 'w')
        self.detail_log = open('../data/detail_log', 'w')

    def training(self):
        print('\nStart training ... \n')
        training_positive_reward_count = 0
        training_start_time = time.time()
        testing_time = 0

        state, reward, terminal = self.framework.new_dialogue()
        for self.step in tqdm(range(self.max_step)):
        # while self.step < self.max_step:
            if self.step % self.evaluate_step == 0 and self.step > self.learn_start:
                self.run_test_cnt += 1
                testing_time = self.test()
                # 如果进行了test,强制结束当前train的dialogue
                # 因为framework和environment里面的step信息是test的了
                terminal = 1

            # self.step += 1

            ask_index, confirm_index = self.QLearner.perceive(state, reward, terminal)

            if reward > 0:
                training_positive_reward_count += 1

            if terminal != 1:
                state, reward, terminal = self.framework.step_dialogue(ask_index, confirm_index)
            else:
                state, reward, terminal = self.framework.new_dialogue()

            if self.step % self.train_report_step == 0 and self.step > self.learn_start:
                training_time = time.time() - training_start_time - testing_time

                # estimate end time
                estimate_time = ''
                left_time = (self.max_step - self.step) / self.train_report_step * training_time
                hour = left_time / 3600
                minute = left_time % 3600 / 60
                second = left_time % 3600 % 60
                estimate_time += (' %d hours' % hour) if hour > 0 else ''
                estimate_time += (' %d minutes' % minute) if minute > 0 else ''
                estimate_time += (' %d s' % second)

                # other report
                print('\nSteps: %d, positive reward count: %d, training time: %ds, '
                      'training rate: %.2ffps, now epsilon: %.2f, left training time: %s\n' %
                      (self.step, training_positive_reward_count, training_time,
                       self.train_report_step / training_time,
                       self.QLearner.ep, estimate_time))
                # reset time variables
                training_positive_reward_count = 0
                testing_time = 0
                training_start_time = time.time()

        self.reward_log.close()
        self.test_log.close()
        self.detail_log.close()

    def test(self):
        testing_start = time.time()
        print('\nTesting starts ...\n')

        total_reward = 0.0
        total_episodes = 0
        episode_reward = 0
        positive_reward_count = 0

        # add more detail on 2016.09.02
        total_length = 0.0
        episode_length = 0
        total_success = 0
        success_dict = defaultdict(int)

        error_times = 0

        # step = 0
        state, reward, terminal = self.framework.new_dialogue(self.test_log)
        for step in tqdm(range(self.test_step)):
            episode_length += 1
            ask_index, confirm_index = self.QLearner.perceive(state, reward, terminal,
                                                         testing=True, testing_ep=0.05)

            state, reward, terminal = self.framework.step_dialogue(ask_index, confirm_index, self.test_log)

            episode_reward += reward
            if reward > 0:
                positive_reward_count += 1
                total_success += 1
                success_dict[error_times] += 1

            if reward < -0.05:
                error_times += 1

            if terminal == 1:
                total_reward = total_reward + episode_reward
                total_length = total_length + episode_length
                total_episodes += 1
                #print("episode reward: %f, total reward: %f" % (episode_reward, total_reward))
                #print("total episodes: %d" % total_episodes)
                
                # reset params
                episode_reward = 0
                episode_length = 0
                error_times = 0
                # restart dialogue
                state, reward, terminal = self.framework.new_dialogue(self.test_log)

        print('\nTesting ends ...\n')

        avg_reward = total_reward / max(1.0, float(total_episodes))
        self.reward_log.write('%.6f\n' % avg_reward)
        self.reward_log.flush()

        # self.QLearner.compute_validation_data()

        if avg_reward > self.max_avg_reward:
            self.max_avg_reward = avg_reward
            self.best_network = self.QLearner.output_network
            # save best network
            print('best network')
        test_time = time.time() - testing_start

        # report test
        print('\nSteps: %d, avg reward: %.2f, testing time: %ds, testing rate: %.2f, '
              'total episode: %d, total positive reward: %d, completion rate: %2f' %
              (self.step, total_reward / float(total_episodes), test_time,
               self.test_step / test_time, total_episodes,
               positive_reward_count, positive_reward_count / float(total_episodes)))
        print('Status: replay memory size: %d, replay insert index: %d\n' %
              (self.QLearner.replay_memory.record_size,
               self.QLearner.replay_memory.index))

        # dump res
        # add on 2016.09.02
        res = {'total_reward': total_reward, 'total_episodes': total_episodes, 'positive_reward_count': positive_reward_count, 'total_length': total_length, 'total_success': total_success, 'success_dict': success_dict, 'id': self.run_test_cnt}
        self.detail_log.write('%s\n' % json.dumps(res, ensure_ascii=False))
        self.detail_log.flush()
        print(res)

        return int(test_time)
