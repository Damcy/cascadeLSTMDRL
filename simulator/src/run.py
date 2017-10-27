#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import sys
import socket
import json
import environment


def main():
    slots = eval(sys.argv[3])
    print("slots: ", slots)
    print(type(slots))
    en = environment.Environment(slots)

    pre_id = -1

    host = 'localhost'
    receive_port = int(sys.argv[1])
    send_port = int(sys.argv[2])

    receive_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_s.bind(('', receive_port))

    send_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print('正在等待接入...')
    request_no = 0
    while True:
        data, address = receive_s.recvfrom(1024)
        if request_no % 1000 < 20:
            print('Received:', data, 'from', address)
        request_no += 1
        str_to_dict = json.loads(data.decode())

        if str_to_dict['id'] == pre_id:
            state = en.react("getAgain", str_to_dict['ask'], str_to_dict['confirm'])
        else:
            state = en.react(str_to_dict['action'], str_to_dict['ask'], str_to_dict['confirm'])

        pre_id = str_to_dict['id']

        # return data
        send_s.sendto(json.dumps(state, ensure_ascii=False).encode(), (host, send_port))


if __name__ == '__main__':
    main()
