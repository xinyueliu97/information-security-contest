# -*- coding: utf-8 -*-

import pickle
import os
import numpy
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import metrics

def train():
	start = time.time()
	dy_feature_matrix = {}
        dy_feature = {}
        for each in os.listdir('/home/ubuntu/Code/dataset/dy_feature'):
            path = os.path.join('/home/ubuntu/Code/dataset/dy_feature/', each)
            dy_feature_matrix = pickle.load(open(path, 'rb'))
            print type(dy_feature_matrix)
	    print dy_feature_matrix
            if len(dy_feature) == 0:
		dy_feature = dy_feature_matrix
            else:
                dy_feature = np.append(dy_feature,dy_feature_matrix,axis=0) 
	    print dy_feature

        pickle.dump(dy_feature_matrix, open('/home/ubuntu/Code/dataset/dy_feature_matrix', 'wb'))

	classification = pickle.load(open('/home/ubuntu/Code/dataset/dy_classification', 'rb'))

	mlp = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(10, 10,5,5,2))

	mlp.fit(dy_feature, classification)  
        
        kf = KFold(len(dy_feature), n_folds=5, shuffle = True)
	print "Cross Validating..."
        predicted = cross_validation.cross_val_predict(mlp, dy_feature, classification, cv=kf)
        print "Confusion matrix: "
        print metrics.confusion_matrix(classification, predicted)
        print("Accuracy: %0.3f" % metrics.accuracy_score(classification, predicted))
        print "Precision: "
        print metrics.precision_score(classification, predicted, average=None)
        print "Recall: "
        print metrics.recall_score(classification, predicted, average=None)
        print "F1 "
        print metrics.f1_score(classification, predicted, average=None)
        print "learning with MLP done.\n"                     
        
	pickle.dump(mlp, open('/home/ubuntu/Code/dataset/dymodel_final', 'wb'))  # 固化训练结果

	print 'time :%f'% (time.time() - start)


if __name__ == '__main__': 
	train()
