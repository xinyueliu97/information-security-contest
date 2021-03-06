# -*- coding: utf-8 -*-

import sys
sys.path.append("..")
sys.path.append("/home/ubuntu/DroidBox/")

import csv
import os
import pickle
import numpy
import re
import shutil


from dy_analyse import *
from sklearn.neural_network import MLPClassifier


def dyrun(decode_path):
	#decode_path = os.path.join(out_path, filename[:-4])

	mlp = pickle.load(open('/home/ubuntu/Code/dataset/dymodel_final', 'rb'))
	shutil.rmtree('/home/ubuntu/DroidBox/output')  
	os.mkdir('/home/ubuntu/DroidBox/output') 

	dy_vector = run(decode_path,'40',1) 

	expect = mlp.predict(dy_vector)

	score = '%.0f' % (mlp.predict_proba(dy_vector)[0][1] * 100)
        #To show the result of dy_analyse,finally combine!!
	img1 = int(score) / 10
    	img2 = int(score) % 10
    	print ("risk score: %s" % score)

    	if expect == 1:
            result = u"预测该应用是 恶意应用"
    	else:
       	    result = u"预测该应用是 良性应用"
        print result

	return expect,score

if __name__ == '__main__': 
     run('/home/ubuntu/apifordy/')
