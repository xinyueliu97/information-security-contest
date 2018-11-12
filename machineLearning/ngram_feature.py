#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import os
import pickle
from infoExtract import ngram
import numpy as np
import time


class NgramFeatureExtract:
    def __init__(self, path):
        """
        初始化
        :param path: path为/home/project/apks/apks
        """
        self.root_path = path
        self.app = dict()
        self.count = 0

    def __find_all_app(self, relative_path):
        # full path is /home/project/apks/apks/ben_code/
        full_path = os.path.join(self.root_path, relative_path)
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
                    if os.path.exists(os.path.join(each_path, 'smali/')):
                        # extra the ngram of apk
                        print '[+]analysing:'+each_path
                        ngram_feature, ngram_names = ngram.ml_ngram_feature(each_path)
                        self.app[self.count] = ngram_feature
                        # the apk num add 1
                        self.count += 1
                        if self.count % 10 == 0 and self.count != 0:  # 每10个数据返回一次
                            print 'app count %d' % self.count
                            yield self.count
                    else:
                        stack.append(each_path)

        if self.count % 10 != 0:
            print 'app count %d' % self.count
            yield self.count

    def run(self):
        ben_app_num = 0
        for count in self.__find_all_app('ben_code/'):
            ben_app_num = count
            api_feature_matrix = self.app  #.values()
            pickle.dump(api_feature_matrix, open('dataset/ngram_feature/ngram_feature_matrix_%d' % count, 'wb'))  # 固化特征矩阵
            self.app = dict()

        for count in self.__find_all_app('mal_code/'):
            api_feature_matrix = self.app #.values()
            pickle.dump(api_feature_matrix, open('dataset/ngram_feature/ngram_feature_matrix_%d' % count, 'wb'))  # 固化特征矩阵
            self.app = dict()

        classification = np.ones(self.count, dtype=np.int)  # 分类结果
        for i in range(ben_app_num):
            classification[i] = 0
        pickle.dump(classification, open('dataset/classification', 'wb'))  # 固化分类结果

if __name__ == '__main__':
    # 平均一个apk 1s
    start = time.time()
    ROOT = r'/home/project/apks/apks'
    NgramFeatureExtract(ROOT).run()
    print 'time :%f'% (time.time() - start)