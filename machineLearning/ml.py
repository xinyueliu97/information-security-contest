# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy
from rbm import test_rbm
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import time
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def mic(x,y):
	m = MINE()
	m.compute_score(x,y)
	return (m.mic(),0.5)

def train():
    # if os.path.exists('dataset/per_feature_matrix'):
    #     per_feature_matrix = pickle.load(open('dataset/per_feature_matrix', 'rb'))
    # else:
    start = time.time()
    print "extracting feature matrix..."
    if 1:
        per_feature_matrix = {}
        for each in os.listdir('dataset/per_feature'):
            path = os.path.join('dataset/per_feature/', each)
            per_feature_matrix = dict(pickle.load(open(path, 'rb')), **per_feature_matrix)
        per_feature_matrix = per_feature_matrix.values()
        pickle.dump(per_feature_matrix, open('dataset/per_feature_matrix', 'wb'))

    # if os.path.exists('dataset/api_feature_matrix'):
    #     api_feature_matrix = pickle.load(open('dataset/api_feature_matrix', 'rb'))
    # else:
    if 1:
        api_feature_matrix = {}
        for each in os.listdir('dataset/api_feature'):
            path = os.path.join('dataset/api_feature/', each)
            api_feature_matrix = dict(pickle.load(open(path, 'rb')), **api_feature_matrix)
        api_feature_matrix = api_feature_matrix.values()
        pickle.dump(api_feature_matrix, open('dataset/api_feature_matrix', 'wb'))

    # if os.path.exists('dataset/ngram_feature_matrix'):
    #     ngram_feature_matrix = pickle.load(open('dataset/ngram_feature_matrix', 'rb'))
    # else:
    if 1:
        ngram_feature_matrix = {}
        for each in os.listdir('dataset/ngram_feature'):
            path = os.path.join('dataset/ngram_feature/', each)
            ngram_feature_matrix = dict(pickle.load(open(path, 'rb')), **ngram_feature_matrix)
        ngram_feature_matrix = ngram_feature_matrix.values()
        pickle.dump(ngram_feature_matrix, open('dataset/ngram_feature_matrix', 'wb'))

    classification = pickle.load(open('dataset/classification', 'rb'))
    if per_feature_matrix is not None and api_feature_matrix is not None and ngram_feature_matrix is not None:
        feature_matrix = _concatenate(per_feature_matrix, api_feature_matrix, ngram_feature_matrix)
    elif per_feature_matrix is not None:
        feature_matrix = per_feature_matrix
    elif api_feature_matrix is not None:
        feature_matrix = api_feature_matrix
    elif ngram_feature_matrix is not None:
        feature_matrix = ngram_feature_matrix
    else:
        return
    print "extracting feature matrix done."
    print "处理前样本总数:%d"%len(feature_matrix)

    #print len(feature_matrix)
    #print len(classification)

    features = 400
    fsmodel = SelectKBest(chi2, k=features)
    raw_feature_matrix = feature_matrix
    feature_matrix = fsmodel.fit_transform(feature_matrix, classification)

    pickle.dump(fsmodel, open('dataset/fsmodel', 'wb'))

    features = 300
    svc = SVC(kernel="linear",C=1)
    fsmodel2 = RFE(estimator=svc,n_features_to_select = features,step = 1)
   
    #########################    DEBUG    ############################
    #classification = classification[7:]
    ##################################################################
    feature_matrix = fsmodel2.fit_transform(feature_matrix, classification)
    
    pickle.dump(fsmodel2, open('dataset/fsmodel2', 'wb'))
	
    #########################    DEBUG    ############################
    b_s = 5			#改这里也要改dl.py里面的默认值
    length = len(feature_matrix)
    feature_matrix = feature_matrix[length%b_s:]
    raw_feature_matrix = raw_feature_matrix[length%b_s:]
    classification = classification[length%b_s:]
    print "处理后样本总数:%d"%len(feature_matrix)
    ##################################################################
	
	#########################    DEBUG    ############################
    fs_vec = []
    for i in range(len(raw_feature_matrix[0])):
        fs_vec.append(i)			#构造值等于编号的特殊向量
	
    fs_vec = fsmodel.transform(fs_vec)
    #print fs_vec
    fs_vec = fsmodel2.transform(fs_vec)
    #print fs_vec
	
    feature_matrix_dl = [x for x in range(len(raw_feature_matrix))]
    for i in range(len(feature_matrix_dl)):
        feature_matrix_dl[i] = [x for x in range(len(raw_feature_matrix[0]) - features)]
    temp = 0
    for i in range(len(raw_feature_matrix[0])):
        if i not in fs_vec:
            print "第%d列特征没有选用" % i
            for j in range(len(feature_matrix_dl)):
                feature_matrix_dl[j][temp] = raw_feature_matrix[j][i]
            temp = temp + 1
			
    #print "行数%d" % len(feature_matrix_dl)
    #print "列数%d" % len(feature_matrix_dl[0])
    #print feature_matrix_dl
	
    ##################################################################
    #hiddeny, da = test_dA(feature_matrix_dl, len(feature_matrix_dl[0]))
    # hiddeny2, test = test_dA(feature_matrix,len(feature_matrix[0]), batch_size=6, da_object = da)
    hiddeny, da = test_rbm(feature_matrix_dl, len(feature_matrix_dl[0]))
    #print len(feature_matrix)
    print "浅度特征数：%d"%len(feature_matrix[0])
    #print len(hiddeny)
    print "深度特征数：%d"%len(hiddeny[0])
    # print (hiddeny == hiddeny2).all()


    #固化深度训练器
    pickle.dump(da, open('dataset/rbmmodel', 'wb')) 

    # 深度特征融合
    feature_matrix = numpy.concatenate((feature_matrix, hiddeny), axis=1)

    Z = []
    count = 0
    for i in feature_matrix:
        Z.append([])
        for j in i:
            Z[count].append(j)

        count += 1

    feature_matrix = Z

    # print feature_matrix

    Z = []
    for i in classification:
        Z.append(int(i))

    classification = Z





    if 1:
        per_feature_matrix2 = {}
        for each in os.listdir('test/per_feature'):
            path = os.path.join('test/per_feature/', each)
            per_feature_matrix2 = dict(pickle.load(open(path, 'rb')), **per_feature_matrix2)
        per_feature_matrix2 = per_feature_matrix2.values()
        pickle.dump(per_feature_matrix2, open('test/per_feature_matrix', 'wb'))

    # if os.path.exists('dataset/api_feature_matrix'):
    #     api_feature_matrix = pickle.load(open('dataset/api_feature_matrix', 'rb'))
    # else:
    if 1:
        api_feature_matrix2 = {}
        for each in os.listdir('test/api_feature'):
            path = os.path.join('test/api_feature/', each)
            api_feature_matrix2 = dict(pickle.load(open(path, 'rb')), **api_feature_matrix2)
        api_feature_matrix2 = api_feature_matrix2.values()
        pickle.dump(api_feature_matrix2, open('test/api_feature_matrix', 'wb'))

    # if os.path.exists('dataset/ngram_feature_matrix'):
    #     ngram_feature_matrix = pickle.load(open('dataset/ngram_feature_matrix', 'rb'))
    # else:
    if 1:
        ngram_feature_matrix2 = {}
        for each in os.listdir('test/ngram_feature'):
            path = os.path.join('test/ngram_feature/', each)
            ngram_feature_matrix2 = dict(pickle.load(open(path, 'rb')), **ngram_feature_matrix2)
        ngram_feature_matrix2 = ngram_feature_matrix2.values()
        pickle.dump(ngram_feature_matrix2, open('test/ngram_feature_matrix', 'wb'))

    classification2 = pickle.load(open('test/classification', 'rb'))
    if per_feature_matrix2 is not None and api_feature_matrix2 is not None and ngram_feature_matrix2 is not None:
        feature_matrix2 = _concatenate(per_feature_matrix2, api_feature_matrix2, ngram_feature_matrix2)
    elif per_feature_matrix2 is not None:
        feature_matrix2 = per_feature_matrix2
    elif api_feature_matrix2 is not None:
        feature_matrix2 = api_feature_matrix2
    elif ngram_feature_matrix2 is not None:
        feature_matrix2 = ngram_feature_matrix2
    else:
        return
    print "extracting feature matrix done."
    print "处理前样本总数:%d"%len(feature_matrix2)

    #print len(feature_matrix)
    #print len(classification)

    features = 400
    fsmodel2 = SelectKBest(chi2, k=features)
    raw_feature_matrix2 = feature_matrix2
    feature_matrix2 = fsmodel.fit_transform(feature_matrix2, classification2)

    features2 = 300
    svc = SVC(kernel="linear",C=1)
    fsmodel2 = RFE(estimator=svc,n_features_to_select = features2,step = 1)
    feature_matrix2 = fsmodel2.fit_transform(feature_matrix2, classification2)
    
	
    #########################    DEBUG    ############################
    b_s = 5			#改这里也要改dl.py里面的默认值
    length = len(feature_matrix2)
    feature_matrix2 = feature_matrix2[length%b_s:]
    raw_feature_matrix2 = raw_feature_matrix2[length%b_s:]
    classification2 = classification2[length%b_s:]
    print "处理后样本总数:%d"%len(feature_matrix2)
    ##################################################################
	
	#########################    DEBUG    ############################
    fs_vec2 = []
    for i in range(len(raw_feature_matrix2[0])):
        fs_vec2.append(i)			#构造值等于编号的特殊向量
	
    fs_vec2 = fsmodel.transform(fs_vec2)
    #print fs_vec
    fs_vec2 = fsmodel2.transform(fs_vec2)
    #print fs_vec
	
    feature_matrix_dl2 = [x for x in range(len(raw_feature_matrix2))]
    for i in range(len(feature_matrix_dl2)):
        feature_matrix_dl2[i] = [x for x in range(len(raw_feature_matrix2[0]) - features2)]
    temp = 0
    for i in range(len(raw_feature_matrix2[0])):
        if i not in fs_vec2:
            print "第%d列特征没有选用" % i
            for j in range(len(feature_matrix_dl2)):
                feature_matrix_dl2[j][temp] = raw_feature_matrix2[j][i]
            temp = temp + 1
			
    hiddeny2, da = test_rbm(feature_matrix_dl2, len(feature_matrix_dl2[0]))
    #print len(feature_matrix)
    print "浅度特征数：%d"%len(feature_matrix2[0])
    #print len(hiddeny)
    print "深度特征数：%d"%len(hiddeny2[0])
    # print (hiddeny == hiddeny2).all()

    # 深度特征融合
    feature_matrix2 = numpy.concatenate((feature_matrix2, hiddeny2), axis=1)

    Z = []
    count = 0
    for i in feature_matrix2:
        Z.append([])
        for j in i:
            Z[count].append(j)

        count += 1

    feature_matrix2 = Z

    # print feature_matrix

    Z = []
    for i in classification2:
        Z.append(int(i))

    classification2 = Z

    
    '''
    kf = KFold(len(feature_matrix2), n_folds=5, shuffle = True)
    print "\nlearning with RF..."
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=10)
    rf.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    scores = cross_validation.cross_val_score(rf, feature_matrix2, classification2, cv=kf)
    print scores
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    print "learning with RF done.\n"
    pickle.dump(rf, open('dataset/model', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
	
    print "learning with GBDT..."
    gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0,
    max_depth=100, min_samples_split=10, random_state=0)
    gbdt.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    scores = cross_validation.cross_val_score(gbdt, feature_matrix2, classification2, cv=kf)
    print scores
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    print "learning with GBDT done.\n"
    pickle.dump(gbdt, open('dataset/model2', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
	
    print "learning with AdaBoost..."
    ada = AdaBoostClassifier(n_estimators=300)
    ada.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    scores = cross_validation.cross_val_score(ada, feature_matrix2, classification2, cv=kf)
    print scores
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    print "learning with AdaBoost done.\n"
    pickle.dump(ada, open('dataset/model3', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
	
    print "learning with LogisticRegression..."
    lr = LogisticRegression()
    lr.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    scores = cross_validation.cross_val_score(lr, feature_matrix2, classification2, cv=kf)
    print scores
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    print "learning with LogisticRegression done.\n"
    pickle.dump(lr, open('dataset/model4', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
    
    kf = KFold(len(feature_matrix2), n_folds=5, shuffle = True)
    print "\nlearning with RF..."
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=10)
    rf.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(rf, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    print "learning with RF done.\n"
    pickle.dump(rf, open('dataset/model', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
	
    print "learning with GBDT..."
    gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0,
    max_depth=100, min_samples_split=10, random_state=0)
    gbdt.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(gbdt, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    print "learning with GBDT done.\n"
    pickle.dump(gbdt, open('dataset/model2', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
	
    print "learning with AdaBoost..."
    ada = AdaBoostClassifier(n_estimators=300)
    ada.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(ada, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    print "learning with AdaBoost done.\n"
    pickle.dump(ada, open('dataset/model3', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
	
    print "learning with LogisticRegression..."
    lr = LogisticRegression()
    lr.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(lr, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    print "learning with LogisticRegression done.\n"
    pickle.dump(lr, open('dataset/model4', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
    '''
    '''
    kf = KFold(len(feature_matrix2), n_folds=5, shuffle = True)
    print "\nlearning with SVC..."
    slffork=SVC(kernel='rbf',probability = True)
    slffork.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(slffork, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    print "learning with SVC done.\n"
    pickle.dump(slffork, open('dataset/model2', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
    '''
    '''
    print "learning with BaggingClassifier..."
    kf = KFold(len(feature_matrix2), n_folds=5, shuffle = True)
    baggingfork = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5,max_features=0.5)
    baggingfork.fit(feature_matrix2, classification2)
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(baggingfork, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    print "learning with BaggingClassifier done.\n"
    pickle.dump(baggingfork, open('dataset/model2', 'wb'))  # 固化训练结果
    #print 'time :%f'% (time.time() - start)
    '''
    '''kf = KFold(len(feature_matrix2), n_folds=5, shuffle = True)'''
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=10)
    gbdt = GradientBoostingClassifier(n_estimators=300,learning_rate=1.0,
    max_depth=100,min_samples_split=10, random_state=0)
    ada = AdaBoostClassifier(n_estimators=300)
    #slf1=SVC(kernel='rbf',probability = True)
    bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5,max_features=0.5)  
   
	
    print "learning with Voting Classifier..."
    vc = VotingClassifier(estimators=[('rf', rf), ('ada', ada), ('bagging', bagging),('gbdt',gbdt)], voting='soft', weights=[1.5, 1.5, 1.3, 1.5])
    vc.fit(feature_matrix, classification)
    '''
    print "Cross Validating..."
    predicted = cross_validation.cross_val_predict(vc, feature_matrix2, classification2, cv=kf)
    print "Confusion matrix: "
    print metrics.confusion_matrix(classification2, predicted)
    print("Accuracy: %0.3f" % metrics.accuracy_score(classification2, predicted))
    print "Precision: "
    print metrics.precision_score(classification2, predicted, average=None)
    print "Recall: "
    print metrics.recall_score(classification2, predicted, average=None)
    print "F1 "
    print metrics.f1_score(classification2, predicted, average=None)
    '''
    print "learning with Ensemble Classifier done.\n"
    pickle.dump(vc, open('dataset/model_final', 'wb'))  # 固化训练结果
    print 'time :%f'% (time.time() - start)
    


def _concatenate(a, b, c):
    """

    :param a: ndarray
    :param b: ndarray
    :return: ndarray
    """
    feature_matrix = numpy.zeros(shape=(len(a), len(a[0]) + len(b[0]) + len(c[0])))  # 特征矩阵，目测是稀疏矩阵，暂时使用numpy array

    for i in range(len(a)):
        for j in range(len(a[0])):
            feature_matrix[i][j] = a[i][j]
        for j in range(len(b[0])):
            feature_matrix[i][j + len(a[0])] = b[i][j]
        for j in range(len(c[0])):
            feature_matrix[i][j + len(a[0]) + len(b[0])] = c[i][j]
    return feature_matrix

