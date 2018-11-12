# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os

__manifest = dict()

def run(apks_path):
    decode_all_apks(apks_path, 'ben')
    decode_all_apks(apks_path, 'mal')
    pass


def decode_all_apks(apks_path, apks_type):
    path = os.path.join(apks_path, apks_type)
    stack = list()
    stack.append(path)
    while len(stack) > 0:
        path = stack.pop()
        for path1 in os.listdir(path):
            full_path = os.path.join(path, path1)
            extension = os.path.splitext(path1)[1]
            filename = os.path.splitext(path1)[0]
            if os.path.isdir(full_path):
                stack.append(full_path)
            elif extension == '.APK' or extension == '.apk':
                # 找到所有的apk文件，反编译之
                if apks_type == 'ben':
                    out_path = apks_path + '/ben_code/' + filename
                else:
                    out_path = apks_path + '/mal_code/' + filename
                cmd = 'apktool d "' + full_path + '" -o ' + '"'+out_path+'"'
                os.system(cmd)  # decode apk


