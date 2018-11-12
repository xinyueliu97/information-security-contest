# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

import os
import numpy as np
import pickle
from infoExtract import api
import time

ALLOWED_EXTENSIONS = set(['smali'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


class ApiFeatureExtract:
    def __init__(self, path):
        self.root_path = path
        self.count = 0
        self.app = dict()
        if os.path.exists('dataset/app_list_ben'):
            self.ben_list = pickle.load(open('dataset/app_list_ben', 'rb'))
            # self.ben_count = len(self.ben_list)
        else:
            print '[-]extract ngram feature : cannot find ben app file data'
            exit()
        if os.path.exists('dataset/app_list_mal'):
            self.mal_list = pickle.load(open('dataset/app_list_mal', 'rb'))
            # self.mal_list = len(self.mal_list)
        else:
            print '[-]extract ngram feature : cannot find mal app file data'
            exit()

    def run(self):
        ben_app_num = 0
        os.system('rm -rf dataset/api_feature/*')
        for count in self.__find_all_app('ben_code'):
            ben_app_num = count
            api_feature_matrix = self.app #.values()
            pickle.dump(api_feature_matrix, open('dataset/api_feature/api_feature_matrix_%d' % count, 'wb'))  # 固化特征矩阵
            self.app = dict()

        for count in self.__find_all_app('mal_code'):
            api_feature_matrix = self.app  # .values()
            pickle.dump(api_feature_matrix, open('dataset/api_feature/api_feature_matrix_%d' % count, 'wb'))  # 固化特征矩阵
            self.app = dict()

        classification = np.ones(self.count, dtype=np.int)  # 分类结果
        for i in range(ben_app_num):
            classification[i] = 0
        pickle.dump(classification, open('dataset/classification', 'wb'))  # 固化分类结果

    def __find_all_app(self, app_type):
        if app_type == 'ben_code':
            app_list = self.ben_list
        elif app_type == 'mal_code':
            app_list = self.mal_list

        for each_path in app_list:
            print '[+]analysing:'+each_path
            api_feature, api_names = api.ml_api_feature(each_path)
            self.app[self.count] = api_feature
            # the apk num add 1
            self.count += 1
            if self.count % 10 == 0 and self.count != 0:  # 每10个数据返回一次
                print 'app count %d' % self.count
                yield self.count
        if self.count % 10 != 0:
            print 'app count %d' % self.count
            yield self.count

if __name__ == '__main__':
    start = time.time()
    ROOT = r'/home/project/apks/apks/'
    ApiFeatureExtract(ROOT).run()
    print 'time :%f'% (time.time() - start)