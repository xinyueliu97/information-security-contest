# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import csv
import os
import pickle
import numpy
import re
import shutil
import dy_analyse1
import g_predict
import dy_predict
import json

from rbm import test_rbm
from infoExtract import permission, api, ngram, component
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression





def metaInfo(file_path, decode_path):
    icon = None
    meta = dict()
    meta['File Md5'] = os.path.split(file_path)[-1][:-4]
    meta['File Size'] = '%.3fKB' % (os.path.getsize(file_path)/1024.0)
    with open(os.path.join(decode_path, 'AndroidManifest.xml')) as manifest:
        content = manifest.read()
        meta['Package Name'] = re.findall("package=\"(.*?)\"", content)[0]
        icon_name = re.findall("android:icon=\"@drawable/(.*?)\"", content)[0]
    drawable = None
    for i in os.listdir(os.path.join(decode_path,'res')):
        if 'drawable' in i and os.path.exists(os.path.join(decode_path, 'res', i, icon_name+'.png')):
            drawable = i
            break
    if drawable:
        icon_old_path = os.path.join(decode_path, "res", drawable, icon_name + '.png')
        shutil.copyfile(icon_old_path, '/home/ubuntu/Code/static/apkimgs/' + meta['File Md5'])
        icon = '/home/ubuntu/Code/static/apkimgs/' + meta['File Md5']
    return icon, meta


def run(filename, out_path):

    decode_path = os.path.join(out_path, filename[:-4])
    cmd = 'apktool d "' + out_path + filename + '" -o ' + decode_path
    os.system(cmd)  # decode apk
    if os.path.exists(os.path.join(decode_path, "AndroidManifest.xml")) is not True:
        icon = None
        meta = []
        all_permission = []
        all_api = []
        result = u"该应用设有防止反编译机制，无法反编译，请换一个应用"
        return icon, meta, all_permission, all_api, result

    icon, meta = metaInfo(os.path.join(out_path, filename), decode_path)

    #permission_vector, all_permission = permission.ml_per_feature(decode_path)  # return all permission needed
    #api_vector, all_api = api.ml_api_feature(decode_path)  # return all api called
    #ngram_vector, all_ngram = ngram.ml_ngram_feature(decode_path)

    ''' 加载训练模型
    '''
    '''
    rf = pickle.load(open('dataset/model', 'rb'))
    gbdt = pickle.load(open('dataset/model2', 'rb'))
    ada = pickle.load(open('dataset/model3', 'rb'))
    lr = pickle.load(open('dataset/model4', 'rb'))
    '''
    vc = pickle.load(open('dataset/model_final', 'rb'))

    '''构造该应用的特征向量
    '''
    permission_vector, all_permission = permission.ml_per_feature(decode_path)  # return all permission needed
    api_vector, all_api = api.ml_api_feature(decode_path)  # return all api called
    ngram_vector, all_ngram = ngram.ml_ngram_feature(decode_path)
    feature_vector = permission_vector + api_vector + ngram_vector


    '''feature-selection
    '''
    #make list to a vector
    train_set_x = list()
    train_set_x.append(feature_vector)
    raw_train_set_x = train_set_x
    fsmodel = pickle.load(open('dataset/fsmodel', 'rb'))
    #print len(train_set_x)
    fs_vector = fsmodel.transform(train_set_x)

    fsmodel2 = pickle.load(open('dataset/fsmodel2', 'rb'))
    fs_vector = fsmodel2.transform(fs_vector)

    feature_vector = fs_vector[0]
    train_set_x = fs_vector# + fs_vector
    #print train_set_x		#[[]]

	#########################    DEBUG    ############################
    fs_vec = []
    for i in range(len(raw_train_set_x[0])):
        fs_vec.append(i)

    print fs_vec
    fs_vec = fsmodel.transform(fs_vec)
    fs_vec = fsmodel2.transform(fs_vec)
    print fs_vec

    feature_matrix_dl = [x for x in range(len(raw_train_set_x))]
    for i in range(len(feature_matrix_dl)):
        feature_matrix_dl[i] = [x for x in range(len(raw_train_set_x[0]) - len(fs_vec[0]))]
    temp = 0
    for i in range(len(raw_train_set_x[0])):
        if i not in fs_vec:
            #print "第%d列特征没有选用" % i
            for j in range(len(feature_matrix_dl)):
                feature_matrix_dl[j][temp] = raw_train_set_x[j][i]
            temp = temp + 1

    #print "行数%d" % len(feature_matrix_dl)
    #print "列数%d" % len(feature_matrix_dl[0])
    train_set_x = feature_matrix_dl
    #print train_set_x
    train_set_x1 = [train_set_x[0]]
    b_s = 5
    for i in range(1, b_s):
        train_set_x1.append(1)
        train_set_x1[i] = train_set_x[0]
    #print train_set_x1
    train_set_x = train_set_x1
    ##################################################################
    print len(raw_train_set_x)
    print len(fs_vec[0])

    rbm = pickle.load(open('dataset/rbmmodel', 'rb'))
    hiddeny, test = test_rbm(train_set_x, train_set_x_feature_num=len(train_set_x[0]), batch_size=b_s, rbm_object = rbm)
    hiddeny = hiddeny[0]

    feature_vector = numpy.concatenate((feature_vector, hiddeny), axis=0).reshape(1, -1).tolist()
    #print feature_vector
    # print len(feature_vector)

    ''' 预测
    '''
    '''
    #[r1, r2, r3, r4] = [rf.predict(feature_vector), gbdt.predict(feature_vector), ada.predict(feature_vector), lr.predict(feature_vector)]
    [p1, p2, p3, p4] = [rf.predict_proba(feature_vector), gbdt.predict_proba(feature_vector), ada.predict_proba(feature_vector), lr.predict_proba(feature_vector)]
    [w1, w2, w3, w4] = [1, 1, 1, 1]
    #expect = w1 * r1 + w2 * r2 + w3 * r3 + w4 * r4
    expect = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4
    print ("risk score: %0.2f" % (expect[0][1] / (expect[0][0] + expect[0][1])))
    #if expect > 2:
    if expect[0][0] < expect[0][1]:
        result = u"预测该应用是  恶意应用"
    else:
        result = u"预测该应用是  良性应用"
    print result
    #print [r1, r2, r3, r4]
    p = [p1, p2, p3, p4]
    print p
    #print all_permission
    '''
    expect1 = vc.predict(feature_vector)
    score1 = '%.0f' % (vc.predict_proba(feature_vector)[0][1] * 100)
    
    print ("risk score: %s" % score1)
    print filename

    if filename.find("apk"):
        expect2,score22 = g_predict.run(out_path + filename)
        print "expect2 = ",expect2
        
    else:
        expect2,score22 = g_predict.run(out_path + filename + ".apk")
        
    
    
    
    print out_path
    if filename.find("apk"):
        expect3,score3 = dy_predict.dyrun(out_path+filename)
    else:
        expect3,score3 = dy_predict.dyrun(out_path + filename + ".apk")


    if expect2 == -1:
	score = score1*0.8+score3*0.2
        expect = expect1*0.8+expect3*0.2
        if expect<0.5:
            expect = 0
        else:
            expect = 1
    else:
        score2 = '%.0f' % (score22*100)
        print "score1",score1
        print "score2",score2
        score2 = int(score2)
        score1 = int(score1)
        score3 = int(score3)
        print "score2 by int",score2
        print "score3",score3
        score = score1*0.6+score2*0.2+score3*0.2
        print score
        score = int(score)
        print score
        expect = expect1*0.6+expect2*0.2+expect3*0.2
        if expect<0.5:
            expect = 0
        else:
            expect = 1

    img1 = int(score) / 10
    img2 = int(score) % 10

    if expect == 1:
        result = u"预测该应用是 恶意应用"
    else:
        result = u"预测该应用是 良性应用"
    print result

    # -------------------- Permission OUTPUT ---------------------------------------------
    permission_output = {}
    # print all_permission
    with open('/home/ubuntu/Code/data/permission.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if row[0].strip() in all_permission:
                permission_output[row[0].strip()] = {}
                permission_output[row[0].strip()]['Description'] = row[1].strip().decode('utf-8')
                permission_output[row[0].strip()]['ThreatLevel'] = row[2].strip()

    # -----------------        Sensitive Api OUTPUT       ---------------------------------
    sensitive_api_output = {}
    sensitive_api_list = {}
    with open('/home/ubuntu/Code/data/all_sensitive_api.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if row[0] != 'API Name':
                sensitive_api_list[row[0].strip()] = {}
                sensitive_api_list[row[0].strip()]['Description'] = row[1].strip().decode('utf-8')
                sensitive_api_list[row[0].strip()]['ThreatLevel'] = row[2].strip()
    # sensitive api输出结构为二维字典
    # packagename :{ api:{'Description':xxx, ThreatLevel:'xxx' }}
    for each_api in all_api:
        if each_api in sensitive_api_list:
            packagename, api_name = each_api.split('->')
            # print packagename, '#' ,api_name
            # print packagename
            # print api_name
            if packagename not in sensitive_api_output:
                sensitive_api_output[packagename] = {}
            sensitive_api_output[packagename][api_name] = sensitive_api_list[each_api]
    # print sensitive_api_output
    # -------------------------   Component        ----------------------------------------
    component_output =  component.run(decode_path)

    # ----------------------          DY OUTPUT     ---------------------------------------
    dy_path = os.path.join('/home/project/apks/uploads/', meta['File Md5'] + '.dy')
    if not os.path.exists(dy_path):
        behavior = dy_analyse1.dy_demo(os.path.join(out_path, filename), meta['Package Name'])
        #print "bebavior:" + behavior
        #pickle.dump(behavior, open(dy_path, 'wb'))
    else:
        print "ddddddd"
        #behavior = pickle.load(open(dy_path, 'rb'))

    #behavior1 = eval(open('/home/project/out.txt','r').read())
    outpath1 = '/home/ubuntu/DroidBox/output/out1.json'
    behavior = [json.loads(line) for line in open(outpath1)]
    #print behavior
    #behavior = behavior1[0]

    # ------------------------         FILE ACCESS  ----------------------------------------
    file_access = []

    if 'fdaccess' in behavior[0]:
        if behavior[0]['fdaccess']:
            file_bebavior = behavior[0]['fdaccess']
            print file_bebavior
            for each_info in file_bebavior:
                file_access.append([file_bebavior[each_info]['path'],file_bebavior[each_info]['operation'],file_bebavior[each_info]['data']])

    # ------------------------         SMS ACCESS  ----------------------------------------
    sms_access = []

    if 'sendsms' in behavior[0]:
        if behavior[0]['sendsms']:
            sms_bebavior = behavior[0]['sendsms']
            for each_info in sms_bebavior:
                sms_access.append([sms_bebavior[each_info]['number'],sms_bebavior[each_info]['message']])

    # ------------------------         Net ACCESS  ----------------------------------------
    net_send = []
    net_recv = []

    if 'sendnet' in behavior[0]:
        if behavior[0]['sendnet']:
            net_send_bebavior = behavior[0]['sendnet']
            for each_info in net_send_bebavior:
                net_send.append([net_send_bebavior[each_info]['desthost'],net_send_bebavior[each_info]['data'],\
                                 net_send_bebavior[each_info]['operation'],net_send_bebavior[each_info]['destport']])
    if 'recvnet' in behavior[0]:
        if behavior[0]['recvnet']:
            net_recv_bebavior = behavior[0]['recvnet']
            for each_info in net_recv_bebavior:
                net_recv.append([net_recv_bebavior[each_info]['host'],net_recv_bebavior[each_info]['port'],\
                                 net_recv_bebavior[each_info]['data']])

#------------------------------- Sensitive Data operation---------------------------------
    data_operation = []
    sensitive_data_list = {}
    with open('/home/ubuntu/Code/data/dy_feature.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if row[0] != 'Feature':
                sensitive_data_list[row[0].strip()] = {}
                sensitive_data_list[row[0].strip()]['Description'] = row[1].strip().decode('utf-8')
        #print sensitive_api_list
    if 'dataleaks' in behavior[0]:
        if behavior[0]['dataleaks']:
            sen_data_bebavior = behavior[0]['dataleaks']
            print sen_data_bebavior
            for item in sensitive_data_list:
                for each_info in sen_data_bebavior:
                    if item in sen_data_bebavior[each_info]['tag']:
                        data_operation.append([str(sen_data_bebavior[each_info]['tag']).decode('utf-8'),sensitive_data_list[item]['Description'],sen_data_bebavior[each_info]['data']])
     
    print "data_operation:",data_operation

#---------------------------------dataleak-----------------------------------------
    data_leak = []
    dataleak_list = {}
    with open('/home/ubuntu/Code/data/dy_feature.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if row[0] != 'Feature':
                dataleak_list[row[0].strip()] = {}
                dataleak_list[row[0].strip()]['Description'] = row[1].strip().decode('utf-8')
        #print dataleak_list
    flag = 0
    if 'sendnet' in behavior[0]:
        if behavior[0]['sendnet']:
            data_leak_bebavior = behavior[0]['sendnet']
            #print data_leak_bebavior
            for item in dataleak_list:
                if str(item).find('sendnet_')!= -1:
                    flag = 1
                    print item,flag
                    for each_info in data_leak_bebavior:
                        temp = (str(item))[8:]
                      
                        if temp in (str(data_leak_bebavior[each_info]['tag'])):
                            print temp
                            data_leak.append([data_leak_bebavior[each_info]['desthost'],str(data_leak_bebavior[each_info]['tag']).decode('utf-8'),dataleak_list[item]['Description']])
                        break
    #print "data_leak:",data_leak

#--------------------------------enfperm---------------------------------------
    enf_per = []
    if 'enfperm' in behavior[0]:
        if behavior[0]['enfperm']:
            enf_bebavior = behavior[0]['enfperm']
            print enf_bebavior
            for each_info in enf_bebavior:
                enf_per.append([enf_bebavior[each_info]])
    print "enf_per:",enf_per

#-------------------------------------------------------------------------------
                




    return icon, meta, permission_output, sensitive_api_output,component_output, file_access, sms_access, net_send, net_recv, data_operation,data_leak,enf_per,result, img1, img2, expect





if __name__ == '__main__':
    print metaInfo(r'W:\apks\uploads\2beefa525f5ba24ceaad42854aa7ae23.apk', r'W:\apks\uploads\2beefa525f5ba24ceaad42854aa7ae23')
