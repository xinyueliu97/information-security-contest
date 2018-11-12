# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import re
import csv
import binascii
import pickle
from numpy import*
from svmutil import*
import scipy as sp
import numpy as np
import svmutil as st
import pickle
import graph

#filepath:upload apk path
def run(filepath):
    print "filepath"+filepath
    filetype = os.path.splitext(filepath)[1]
    name = filepath.split(filetype)[0]
    gexfpath = name +'.gexf'
    decodepath = name+'.txt'
    print "decodepath:"+decodepath	
    #apk to gexf
    cmd = '/home/ubuntu/androguard-2.0/androgexf.py -i '+filepath+' -o '+ gexfpath
    print cmd
    if os.system(cmd)!=0:
	return -1,-1
    #gexf to decode
    graph._component_extract(gexfpath,decodepath)
    #get the test matrix
    f_matrix = T_matrix(decodepath,1980)
    f_label = [0]
    model=svm_load_model('/home/ubuntu/Code/dataset/gmodel')
    #predict
    [p_label,p_acc,p_val] = svm_predict(f_label,f_matrix,model,'-b 1')
    
    #print p_label,p_acc,p_val
    
    p_labels = int(p_label[0])
    #print "lable = %d" % p_label[0]
    #print "p_labels =",p_labels
    #print "acc= ",p_acc[p_labels]
        
    p_accs = int(p_acc[p_labels])
    #print p_accs

    #print "val = ",p_val[0][p_labels]
    p_vals = int(p_val[0][p_labels])
    #print p_vals

    return p_labels,p_val[0][p_labels]
    

#get the test matrix 
def T_matrix(decodepath,apknum):
    testnum = 1
    train_path = '/home/ubuntu/Code/dataset/decodefile/decode'
    #km = mat(zeros((row,col)),dtype=double)      
    T=mat(zeros((testnum,int(apknum))),dtype=double)    
    for i in range(0,testnum):
	print i
	for j in range(0,apknum):
	    T[i,j]=graph.kernal_G(decodepath,train_path+str(j+1))
    arr = np.arange(1,int(testnum)+1,1).reshape(int(testnum),1)
    Ttrain = np.column_stack((arr, T)).tolist()
    #print Ttrain
    return Ttrain
'''
#output the node_decode file		
def _component_extract(decode_path,writepath):
    all_component = dict()
    all_edges = dict()
    s_packlist = _init_table()
    gephi = '{http://www.gephi.org/gexf}' 
    root = ET.parse(decode_path).getroot()
    #print root
    
    #all_component['nodes'] = defaultdict(list)
    all_component = defaultdict(list)
    all_edges = defaultdict(list)
    for graph in root.findall(gephi+'graph'):
        for nodes in graph.findall(gephi +'nodes'):
            for node in nodes.findall(gephi+'node'):
                id = str(node.get('id'))
                #label = node.get('label')
                for attvalues in node.findall(gephi+'attvalues'):
                    for attvalue in attvalues.findall(gephi+'attvalue'):
                        aid = attvalue.get('id')
                        if (aid=='3'):
                            value = attvalue.get('value')
			    v_decode = value_decode(value,s_packlist)
                            #print value
                            all_component[id].append(v_decode)
			    all_component[id] = clean(all_component[id])
			    #print all_component[id]
         
        for edges in graph.findall(gephi+'edges'):
            for edge in edges.findall(gephi+'edge'):
                 source = edge.get('source')
                 target = edge.get('target')
                 target_code = all_component[source]
                 all_edges[source].append(target_code)
                 #print all_edges[source]
        
        neighbor(all_edges) 
        f= open (writepath,'w')	
        for x in all_component:
            if x in all_edges:
                so = all_component[x]
                so = str(so)
                #print so
                so = dot1(so)
                #print so,temp
                all_component[x] = xor(so,all_edges[x])		
		if all_component[x]!='00000000000' :
		    print >>f,all_component[x]
	
                                
    return all_component

def clean(so):
    so = str(so)
    so = ''.join(so.split('['))
    so = ''.join(so.split(']'))
    so = ''.join(so.split('\''))
    return so

def xor(a,b):
    x = 0
    y = ''
    for x in range(0,len(a)):
        temp = int(a[x:x+1]) ^ int(b[x:x+1])
        temp = str(temp)
        y = y + temp
    return y

def neighbor(lis):
    for x in lis:
        result = '00000000000'
        for y in lis[x]:	    
            y = str(y)
            result = xor(result,str(y))
        lis[x] = result

def dot1(a):
    b = ''
    temp = a[0:1]
    b=a[1:len(a)]+temp
    return b

def _init_table():
    pack_list = []
    with open('/home/ubuntu/Code/data/sensitive_package.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if row[0] != 'Packname':
                pack_list.append(row[0].strip())
    return pack_list

def value_decode(value,s_packlist):
    decode = ''
    for each_pack in s_packlist:
	if (value.upper().find(each_pack.upper())!=-1):
	    decode = decode + '1'
	else:
	    decode = decode + '0'
    return decode

def kernal_G(filename1,filename2):
    g1 = read_Graph(filename1)  
    g2 = read_Graph(filename2)   
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
'''
if __name__ == '__main__':
    filepath = "/home/ubuntu/project/apks/apks/ben/jianyijizhang.apk"  
    print run(filepath)
    
   
    




