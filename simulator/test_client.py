#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import sys
import json
import socket


test_case = [{"action": "new_dialogue", "ask": "nothing", "confirm": "nothing", "id": 1},
             {"action": "action", "ask": "location", "confirm": "nothing", "id": 2},
             {"action": "action", "ask": "time", "confirm": "location", "id": 3},
             {"action": "action", "ask": "nothing", "confirm": "time", "id": 4}]

object = ["time", "duration", "money", "location", "number", "nothing"]


def parse_output(data):
    # json data
    # print('Received:', data.decode(), 'from', addr)
    # format data
    json_to_dict = json.loads(data.decode())
    print("agent:\t", json_to_dict['agent_text'])
    print("user:\t", json_to_dict['user_text'])
    print("reward: ", str(json_to_dict['reward']), "terminal: ", str(json_to_dict['terminal']))


def main():
    host = "localhost"
    send_port = 8080
    receive_port = 8081

    receive_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_s.bind((host, receive_port))

    send_s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    print("Select mode(1 or 2): ")
    print("\t1. auto")
    print("\t2. human input")
    mode = int(input("> "))
    if mode == 1:
        start_index = 0
        while True:
            send_s.sendto(json.dumps(test_case[start_index]).encode(), (host, send_port))
            data, addr = receive_s.recvfrom(1024)
            parse_output(data)
            start_index += 1
            if start_index == len(test_case):
                # start_index = 0
                break
    elif mode == 2:
        id = 1
        while True:
            print("action(new_dialogue/action, default is action): ")
            action = input("> ")
            if action == "new_dialogue":
                to_send = {"action": "new_dialogue", "ask": "nothing", "confirm": "nothing", "id": id}
            else:
                ask = ""
                confirm = ""
                while ask not in object:
                    print("ask(" + '/'.join(object) + "): ")
                    ask = input("> ")
                while confirm not in object:
                    print("confirm(" + '/'.join(object) + "): ")
                    confirm = input("> ")
                to_send = {"action": "action", "ask": ask, "confirm": confirm, "id": id}
            id += 1
            send_s.sendto(json.dumps(to_send).encode(), (host, send_port))
            data, addr = receive_s.recvfrom(1024)
            parse_output(data)
    else:
        print("Error number!")


if __name__ == '__main__':
    main()

