#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import os
import sys
import re
import random
import xml.etree.cElementTree as ET

_DECODE = 'utf-8'
_ENCODE = 'utf-8'


# user generator
class UG(object):
    def __init__(self, args):
        self.slots = args['slots']
        self.template_path = args['setting_path']

        self.inform_template = {}
        self.confirm_template = {}
        self.nothing_template = []
        self.reject_template = []
        self.hello_template = []
        self.load_template()

    def text_response(self, inform="nothing", confirm="nothing", inform_val="nothing", confirm_val="nothing", reject=False, hello=False):
        """
        generate user text according inform, confirm and inform&confirm value, sometimes use reject
        :param inform: inform slot, string
        :param confirm: confirm slot, string
        :param inform_val: inform value
        :param confirm_val: confirm value
        :param reject: reject the user confirm? bool
        :return:
        """
        if hello:
            return self.hello_template[random.randint(0, len(self.hello_template) - 1)]

        if reject:
            # if have reject action, we just care the new inform
            response = [self.reject_template[random.randint(0, len(self.reject_template) - 1)]]
            if inform != "nothing":
                templates = self.inform_template[inform]
                tmp = templates[random.randint(0, len(templates) - 1)]
                response.append(re.sub("XXX", inform_val, tmp))
            return ' '.join(response)
        else:
            # if it's no response or others
            if inform == "nothing" and confirm == "nothing":
                return self.nothing_template[random.randint(0, len(self.nothing_template) - 1)]
            # have confirm or inform
            response = []
            if confirm != "nothing":
                if random.uniform(0, 1) < 0.6 or inform == "nothing":
                    if confirm_val != "nothing":
                        special_template = self.confirm_template['special_confirm'][confirm]
                        confirm_str = special_template[random.randint(0, len(special_template) - 1)]
                        response.append(re.sub("XXX", confirm_val, confirm_str))
                    else:
                        general_template = self.confirm_template['general_confirm']
                        response.append(general_template[random.randint(0, len(general_template) - 1)])
            if inform != "nothing":
                templates = self.inform_template[inform]
                inform_str = templates[random.randint(0, len(templates) - 1)]
                response.append(re.sub("XXX", inform_val, inform_str))
            return ' '.join(response)

    def load_template(self):
        """
        load template from xml
        :return: None
        """
        root = ET.parse(self.template_path)
        user_inform = root.find('user_inform')
        user_confirm = root.find('user_confirm')
        user_nothing = root.find('user_nothing')
        user_reject = root.find('user_reject')
        user_hello = root.find('user_hello')
        self.parse_inform_template(user_inform)
        self.parse_confirm_template(user_confirm)
        self.parse_nothing_template(user_nothing)
        self.parse_reject_template(user_reject)
        self.parse_hello_template(user_hello)

    def parse_inform_template(self, inform_xml):
        for item in self.slots:
            object_template = []
            xml_root = inform_xml.find(item)
            content = xml_root.findall('template')
            for template in content:
                object_template.append(template.text)
            self.inform_template[item] = object_template

    def parse_confirm_template(self, confirm_xml):
        # special confirm
        special_confirm = {}
        special_confirm_xml = confirm_xml.find('special_confirm')
        for item in self.slots:
            object_template = []
            xml_root = special_confirm_xml.find(item)
            content = xml_root.findall('template')
            for template in content:
                object_template.append(template.text)
            special_confirm[item] = object_template
        # general confirm
        general_confirm = []
        general_confirm_xml = confirm_xml.find('general_confirm')
        content = general_confirm_xml.findall('template')
        for template in content:
            general_confirm.append(template.text)

        self.confirm_template['general_confirm'] = general_confirm
        self.confirm_template['special_confirm'] = special_confirm

    def parse_nothing_template(self, nothing_xml):
        content = nothing_xml.findall('template')
        for template in content:
            self.nothing_template.append(template.text)

    def parse_reject_template(self, reject_xml):
        content = reject_xml.findall('template')
        for template in content:
            self.reject_template.append(template.text)

    def parse_hello_template(self, hello_xml):
        content = hello_xml.findall('template')
        for template in content:
            self.hello_template.append(template.text)


def main():
    # here is UG test
    args = {'slots': ["time", "number", "money", "duration", "location"], 'setting_path': "../setting/xml_user.xml"}
    ug = UG(args)
    print(ug.text_response("time", "location", "3/4/2016", "beijing"))
    print(ug.text_response("time", "location", "3/4/2016", "nothing"))
    print(ug.text_response(reject=True))


if __name__ == '__main__':
    main()
