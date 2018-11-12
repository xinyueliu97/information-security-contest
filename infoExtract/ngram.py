#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import itertools

# 控制指令
opcode_dic = {
    'M': '^move',
    'R': '^return',
    'G': '^goto',
    'I': '^if',
    'T': '^[a-z]get',
    'P': '^[a-z]put',
    'V': '^invoke'
}


_ALLOWED_EXTENSIONS = set(['smali'])


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in _ALLOWED_EXTENSIONS


def _init_table():
    table = list('MRGITPV')
    ngram = list(itertools.permutations('MRGITPV', 2))
    ngram += list(itertools.permutations('MRGITPV', 3))

    for i in ngram:
        table.append(''.join(i))
    return table


def _opcode_simplify(smali_file):
    """
    生成简化的指令序列
    :param smali_file: file对象
    :return:简化的执行序列 如 ['M','P','V']
    """
    global opcode_dic
    result_list = []
    for eachLine in smali_file:
        temp = eachLine.strip()
        for each_opcode in opcode_dic:
            if re.search(opcode_dic[each_opcode], temp) is not None:
                result_list.append(each_opcode)
                break
    return result_list


def _opcode_count(smali_file, result):
    """
    统计生成4-gram字典
    :param smali_file: file对象
    :param result:{'M':1,...'MV':1...'MVV':1...}
    :return:{'M':10,...'MV':11...'MVV':12...}
    """

    opcode_list = _opcode_simplify(smali_file)

    for i in xrange(len(opcode_list)):
        if i >= 0:  # 1gram
            if opcode_list[i] not in result:
                result[opcode_list[i]] = 1
            else:
                result[opcode_list[i]] += 1
        if i >= 1:  # 2gram
            gram_2 = opcode_list[i-1] + opcode_list[i]
            if gram_2 not in result:
                result[gram_2] = 1
            else:
                result[gram_2] += 1
        if i >= 2:  # 3gram
            gram_3 = opcode_list[i-2] + opcode_list[i-1] + opcode_list[i]
            if gram_3 not in result:
                result[gram_3] = 1
            else:
                result[gram_3] += 1


def _opcode_feature_extract(decode_path):
    """
    获取文件夹下所有的smali文件并提取opcode 4gram特征
    :param path: apk decode path
    :return:result
    """
    # result = {'1gram': {}, '2gram': {}, '3gram': {}}
    result = {} # 指令序列 -> 出现次数
    smali_path = os.path.join(decode_path, 'smali/')
    dir_list = os.listdir(smali_path)
    if 'android' in dir_list:
        dir_list.remove('android')
    stack = []
    for i in dir_list:
        stack.append(os.path.join(smali_path, i))
    while len(stack) > 0:
        path = stack.pop()
        if os.path.isdir(path):
            for each_path in os.listdir(path):
                temp_path = os.path.join(path, each_path)
                if not os.path.isdir(temp_path):
                    if _allowed_file(temp_path):
                        with open(temp_path, 'r') as f:
                            _opcode_count(f, result)
                else:
                    stack.append(temp_path)
        elif _allowed_file(path):
            with open(path, 'r') as f:
                _opcode_count(f, result)
    return result

def ml_ngram_feature(decode_path):
    """
    为特征列表编号，作为ml的输入，返回从编号0-255的共256个编号好的数据
    :param decode_path: apk decode path
    :return:
    """

    result = _opcode_feature_extract(decode_path)
    table = _init_table()
    ml_feature_list = []  # 按指定顺序排列好的指令特征

    num = float(sum(result.values()))
    for each in table:
        if each in result:
            ml_feature_list.append(result[each]/num)
        else:
            ml_feature_list.append(0.0)

    return ml_feature_list, result.keys()

if __name__ == '__main__':
    PATH = 'E:/0Android/Tools/AndroidKiller/projects/PartOne/Project/'

    ngram = ml_ngram_feature(PATH)
    smali = os.path.join(PATH, 'smali/')
    print ngram
    print len(ngram)


