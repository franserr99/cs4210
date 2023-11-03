#-------------------------------------------------------------------------
# AUTHOR: Francisco Serrano
# FILENAME: svm.py
# SPECIFICATION: Program that builds svm models and hypertunes the parameters.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 45min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
max_accuracy=0
best_param={}

for c_param in c:
    for d_param in degree:
        for k_param in kernel:
            for shape_param in decision_function_shape:
                clf=svm.SVC(C=c_param, degree=d_param, kernel=k_param, decision_function_shape = shape_param)
                #Fit SVM to the training data
                clf.fit(X=X_training,y=y_training)
                #make the SVM prediction for each test sample and start computing its accuracy
                tp=0
                #multi class so its harder to get fn/fp/tn
                #the rest is the sum of tn,fp,fn
                the_rest=0
                
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    prediction=clf.predict([x_testSample])[0]
                    if y_testSample==prediction:
                        tp+=1
                    else:
                        the_rest+=1
                if(tp!=0 and the_rest !=0):
                    accuracy=tp/(tp+the_rest)
                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if(accuracy>max_accuracy):
                    max_accuracy=accuracy
                    best_param['c']=c_param
                    best_param['degree']=d_param
                    best_param['kernel']=k_param
                    best_param['shape']=shape_param
                    print(f"Highest SVM accuracy so far: {max_accuracy}, Parameters: c={c_param}, degree={d_param}, kernel= {k_param}, decision_function_shape = {shape_param}")