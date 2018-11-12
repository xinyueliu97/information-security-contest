# -*- coding: utf-8 -*-
import os
import re
import csv


def ml_api_feature(decode_path):
    sensitive_api = _init_table()
    api = _api_extract(decode_path)
    num = float(sum(api.values()))
    ml_feature_list = []
    for each_api in sensitive_api:
        if each_api in api:
            ml_feature_list.append(api[each_api]/num)
        else:
            ml_feature_list.append(0.0)
    # print api
    # raw_input()
    return ml_feature_list, api.keys()


def _init_table():
    """
    初始化api列表，用于编号和筛选特征api
    :return:
    """
    api_list = []
    with open('data/all_sensitive_api.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if row[0] != 'API Name':
                api_list.append(row[0].strip())
    return api_list


def _api_extract(decode_path):
    smali_path = decode_path + '/smali/'
    # bfs
    api = {}
    for path1 in os.listdir(smali_path):
        if path1 != 'android':
            stack = list()
            full_path = os.path.join(smali_path, path1)
            if os.path.isdir(full_path):
                stack.append(full_path)
            else:
                _regular(full_path, api)
            # dfs
            while len(stack) > 0:
                tmp = stack.pop()
                for path in os.listdir(tmp):
                    next_path = os.path.join(tmp, path)
                    if os.path.isdir(next_path):
                        stack.append(next_path)
                    else:  # all .smali file
                        _regular(next_path, api)

    return api


def _regular(file_path, api):
    f = open(file_path, 'r')
    content = f.read()
    str_list = re.findall('invoke[^\n]*Landroid[^\n]*->[^\n]*', content)
    for item in str_list:
        #  lib
        start = item.find('Landroid')
        end = item.find('->')
        lib_str = item[start + 1:end - 1]
        lib_str = lib_str.replace('/', '.')
        lib_str = lib_str.replace('$', '.')
        # function
        function_str = re.search('->\w+\(', item[end::])
        if function_str:
            fun_name = function_str.group()[2:-1] + '()'
            if api.get(lib_str) is None:
                # api[lib_str] = {}
                # api[lib_str][fun_name] = 1
                api[lib_str+'->'+fun_name] = 1
            else:
                if api[lib_str].get(fun_name) is None:
                    # api[lib_str][fun_name] = 1
                    api[lib_str+'->'+fun_name] = 1
                else:
                    # api[lib_str][fun_name] += 1
                    api[lib_str+'->'+fun_name] += 1
    f.close()


if __name__ == '__main__':
    PATH = ur'W:\apks\apks\ben_code/手机夜视仪_2.10_2012-10'

    table = _init_table()
    print len(table)
    result = ml_api_feature(PATH)
    print result
    print len(result)
