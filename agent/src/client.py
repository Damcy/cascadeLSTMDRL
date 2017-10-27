#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import json
import socket

host = 'localhost'
send_port = 8080
receive_port = 8081

receive_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_s.bind((host, receive_port))

send_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

message_id = 1


def server_interact(action_type, ask='nothing', confirm='nothing', file=None):
    global receive_s
    global send_s
    global message_id
    message_to_send = {'action': action_type,
                       'ask': ask,
                       'confirm': confirm,
                       'id': message_id}
    json_to_dict = {}
    cs_state = 'failed'
    while cs_state == 'failed':
        send_s.sendto(json.dumps(message_to_send).encode(), (host, send_port))
        data, addr = receive_s.recvfrom(1024)
        json_to_dict = json.loads(data.decode())
        cs_state = json_to_dict['state']
        if cs_state == 'failed':
            print("Server Return Error:", json_to_dict['message'])

    message_id += 1

    reward = json_to_dict['reward']
    terminal = json_to_dict['terminal']
    user_text = json_to_dict['user_text']
    agent_text = json_to_dict['agent_text']
    state = agent_text + " # " + user_text + " @"

    if file is not None:
        if action_type == 'new_dialogue':
            file.write('\n############### \nstart new dialogue \n')
        file.write('agent: %s\n' % agent_text)
        file.write('user: %s\n' % user_text)
        file.write('reward: %f\n' % reward)

    return state, reward, terminal


def main():
    pass


if __name__ == '__main__':
    main()
