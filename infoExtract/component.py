# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET


def run(decode_path):
    all_component = _component_extract(decode_path)
    return all_component


def _component_extract(decode_path):
    """找到所有应用具有的权限,保存在all_permisison变量中
    """
    all_component = dict()
    android = '{http://schemas.android.com/apk/res/android}'
    root = ET.parse(decode_path + '/AndroidManifest.xml').getroot()
    all_component['activity'] = {}
    for item in root.iter('activity'):
        name = item.get(android + 'name')
        all_component['activity'][name] = []
        for intent in item.iter('intent-filter'):
            for each in intent:
                if each.get(android + 'name') is not None:
                    all_component['activity'][name].append(each.get(android + 'name'))

    all_component['service'] = {}
    for item in root.iter('service'):
        name = item.get(android + 'name')
        all_component['service'][name] = []
        for intent in item.iter('intent-filter'):
            for each in intent:
                if each.get(android + 'name') is not None:
                    all_component['service'][name].append(each.get(android + 'name'))

    all_component['receiver'] = {}
    for item in root.iter('receiver'):
        name = item.get(android + 'name')
        all_component['receiver'][name] = []
        for intent in item.iter('intent-filter'):
            for each in intent:
                if each.get(android + 'name') is not None:
                    all_component['receiver'][name].append(each.get(android + 'name'))

    all_component['provider'] = {}
    for item in root.iter('provider'):
        name = item.get(android + 'name')
        all_component['provider'][name] = []


    return all_component


if __name__ == '__main__':
    PATH = r'W:/apks/uploads/3e7a3ccf65432d8f21321754aba50e2e'
    print run(PATH)

