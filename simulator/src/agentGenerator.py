#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import os
import sys
import random
import re
import xml.etree.cElementTree as ET

_DECODE = 'utf-8'
_ENCODE = 'utf-8'


# agent generator
class AG(object):
    def __init__(self, args):
        self.slots = args['slots']
        self.template_path = args['setting_path']

        self.ask_template = {}
        self.confirm_template = {}
        self.nothing_template = []
        self.hello_template = []
        self.load_template()

    def text_response(self, ask="nothing", confirm="nothing", confirm_val="nothing", hello=False):
        """
        generate agent text according ask, confirm adn confirm value
        :param ask: ask slot, string
        :param confirm: confirm slot, string
        :param confirm_val: confirm value
        :param hello: hello action? bool
        :return: text, string
        """
        if hello:
            return self.hello_template[random.randint(0, len(self.hello_template) - 1)]

        if ask == 'nothing' and confirm == 'nothing':
            # response nothing
            return self.nothing_template[random.randint(0, len(self.nothing_template) - 1)]
        else:
            response = []
            if confirm != 'nothing':
                templates = self.confirm_template[confirm]
                tmp = templates[random.randint(0, len(templates) - 1)]
                # replace value
                response.append(re.sub("XXX", confirm_val, tmp))
            if ask != 'nothing':
                templates = self.ask_template[ask]
                response.append(templates[random.randint(0, len(templates) - 1)])
            return ' '.join(response)

    def load_template(self):
        """
        load template from xml
        :return: None
        """
        root = ET.parse(self.template_path)
        agent_ask = root.find('agent_ask')
        agent_confirm = root.find('agent_confirm')
        agent_nothing = root.find('agent_nothing')
        agent_hello = root.find('agent_hello')
        self.parse_ask_template(agent_ask)
        self.parse_confirm_template(agent_confirm)
        self.parse_nothing_template(agent_nothing)
        self.parse_hello_template(agent_hello)

    def parse_ask_template(self, ask_xml):
        for item in self.slots:
            object_template = []
            xml_root = ask_xml.find(item)
            content = xml_root.findall('template')
            for template in content:
                object_template.append(template.text)
            self.ask_template[item] = object_template

    def parse_confirm_template(self, confirm_xml):
        for item in self.slots:
            object_template = []
            xml_root = confirm_xml.find(item)
            content = xml_root.findall('template')
            for template in content:
                object_template.append(template.text)
            self.confirm_template[item] = object_template

    def parse_nothing_template(self, nothing_xml):
        content = nothing_xml.findall('template')
        for template in content:
            self.nothing_template.append(template.text)

    def parse_hello_template(self, hello_xml):
        content = hello_xml.findall('template')
        for template in content:
            self.hello_template.append(template.text)


def main():
    # here is AG test
    args = {}
    args['slots'] = ["time", "number", "money", "duration", "location"]
    args['setting_path'] = "../setting/xml_agent.xml"
    ag = AG(args)
    print(ag.text_response("time", "location", "beijing"))
    print(ag.text_response())


if __name__ == '__main__':
    main()
