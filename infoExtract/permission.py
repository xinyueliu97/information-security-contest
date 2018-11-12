# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import csv

__permission = dict()


def run(decode_path):
    all_permission = __permission_extract(decode_path)
    return all_permission


def __permission_extract(decode_path):
    """找到所有应用具有的权限,保存在all_permisison变量中
    """
    all_permission = dict()
    android = '{http://schemas.android.com/apk/res/android}'
    root = ET.parse(decode_path + '/AndroidManifest.xml').getroot()
    for item in root.iter('uses-permission'):
        p = item.get(android + 'name')
        if __permission.get(p):
            all_permission[p] = __permission[p]
    return all_permission


def __read_csv():
    with open('data/permission.csv') as f:
        # with open(url_for('static',filename = 'permission.csv')) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            __permission[row[0]] = row[1]
        #     print row[0]
        #     print row[1]
        #     raw_input()
        # print len(__permission)


def ml_per_feature(decode_path):
    ml_feature_list = []
    permission_description = run(decode_path)
    permission = permission_description.keys()
    all_permission = __permission.keys()
    for each in all_permission:
        if each in permission:
            ml_feature_list.append(1.0)
        else:
            ml_feature_list.append(0.0)
    return ml_feature_list, permission_description

__read_csv()

if __name__ == '__main__':
    PATH = r'/home/project/apks/uploads/3e7a3ccf65432d8f21321754aba50e2e'

    res1, res2 = ml_per_feature(PATH)
    print res1
    print len(res1)
    print res2
    print len(res2)
