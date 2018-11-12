# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import re
import binascii
from numpy import*
from svmutil import*
import scipy as sp
import numpy as np
import svmutil as st
import pickle
import graph

def kernal_G(fileName1,fileName2):
    g1 = read_Graph(fileName1)  
    g2 = read_Graph(fileName2)   
    count = 0
    for a in g1:	
	if a!='00000000000' and a!='[\'00000000000\']':
	    for b in g2:
		b = str(b)
		if a == b:
		    count = count + 1
    return count

#read the node_decode file
def read_Graph(fileName):
    data1 = []
    file = open(fileName,'r')
    for eachLine in file:
	(tmp1,tmp2) = eachLine.split('\n',1)
	data1.append(tmp1)
    return data1

#get the SVM model and save to file
def kernel_matrix(train_path,ben_count,mal_count):
    #generate the kernel matrix 
    train_path = train_path + '/decode'
    apknum = ben_count + mal_count
    M=mat(zeros((int(apknum),int(apknum))),dtype=double)    
    for i in range(0,apknum):
	print train_path+str(i+1)
	for j in range(0,i+1):
	    #print train_path+str(i+1)
	    M[i,j]=M[j,i]=kernal_G(train_path+str(i+1),train_path+str(j+1))
    arr = np.arange(1,int(apknum)+1,1).reshape(int(apknum),1)
    #print np.column_stack((arr, M))
    Mtrain = np.column_stack((arr, M)).tolist()
    pickle.dump(Mtrain, open('/home/ubuntu/Code/dataset/svmmat', 'wb'))
    train_label = [0]*ben_count+[1]*mal_count
    #ben:0 mal:1
    print train_label
    model = svm_train(train_label,Mtrain,'-t 4 -b 1')
    svm_save_model('/home/ubuntu/Code/dataset/gmodel1',model)
    return model

#get the test matrix 
def T_matrix(testnum,apknum):
    train_path = '/home/ubuntu/Code/dataset/decodefile1/decode'
    test_path = '/home/ubuntu/Code/dataset/decodefile1/decode'
    #km = mat(zeros((row,col)),dtype=double)      
    T=mat(zeros((testnum,int(apknum))),dtype=double)    
    for i in range(0,testnum):
	print i
	for j in range(0,apknum):
	    T[i,j]=kernal_G(test_path+str(i+1),train_path+str(j+1))
    arr = np.arange(1,int(testnum)+1,1).reshape(int(testnum),1)
    Ttrain = np.column_stack((arr, T)).tolist()
    return Ttrain

#train_main()     
def train_graph():
    ben_count,mal_count = graph.get_graph_decode()
    print ben_count,mal_count
    train_path = '/home/ubuntu/Code/dataset/decodefile1'
    kernel_matrix(train_path,ben_count,mal_count)
    

if __name__ == '__main__':
    #path of output data
    #train_path = '/media/ubuntu/C45B-9EF4/decodefile/decode'
    #test_path = '/home/ubuntu/Downloads/outfile1/out'
    '''
    #need to count the number of apk 
    apknum = 1980
    testnum = 50

    #kernel_matrix(train_path,apknum)
    #load the model from file
    model=svm_load_model('/home/ubuntu/Code/dataset/gmodel_test')
    
    
    #just for test
    t = T_matrix(testnum,apknum)
    Tlabel = [0]*50
    [p,a,d] = svm_predict(Tlabel,t,model)
    print Tlabel,p,a,d
    '''
    train_graph()


    
    

    




