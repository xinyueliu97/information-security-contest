#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

def find_all_app(apks_path, relative_path):
    root_path = apks_path
    app_list = []
    # full path is /home/project/apks/apks/ben_code/
    full_path = os.path.join(root_path, relative_path)
    stack = list()
    stack.append(full_path)
    while len(stack) > 0:
        path = stack.pop()
        # each file or folder in ben_code/
        for each in os.listdir(path):
            # absolute path
            each_path = os.path.join(path, each)
            # is folder?
            if os.path.isdir(each_path):
                # contains smali folder ?
                if os.path.exists(os.path.join(each_path, 'smali/')) and \
                        os.path.exists(os.path.join(each_path, 'AndroidManifest.xml')):
                    print '[+]find app:%s' % each_path
                    app_list.append(each_path)
                    # the apk num add 1
                else:
                    stack.append(each_path)
    if 'ben' in relative_path:
        dump_file_name = 'ben'
    else:
        dump_file_name = 'mal'

    pickle.dump(app_list, open('dataset/app_list_%s' % dump_file_name, 'wb'))
    print '[+]app count:%d' % len(app_list)
    return app_list

if __name__ == '__main__':
    ben_list = find_all_app('ben_code/')
    mal_list = find_all_app('mal_code/')

    print len(ben_list)
    print len(mal_list)