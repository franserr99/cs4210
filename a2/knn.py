#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db= []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
        if(i>0):
          db.append(row)
      
n=len(db[0])
misclassifications=0
#loop your data to allow each instance to be your test set
for i,data in enumerate(db):

    #add the training features to the 2D array X
    # removing the instance that will be used for testing in this iteration. 
    # For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    print(f"here is the data record {data}")
    X =[]
    for j,record in enumerate(db):   
        print(record)
        if(i!=j):
          features=[]
          features.append(int(record[0]))
          features.append(int(record[1]))
          X.append(features)
    print(X)
    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    label_num={
        '+':1,
        '-':2
    }
    Y =[]
    for j,record in enumerate(db):
        if(i!=j):
          Y.append(label_num[record[n-1]])
    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample =data


    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    class_predicted = clf.predict([[int(testSample[0]),int(testSample[1])]])[0]
    #--> add your Python code here
    true_label=label_num[testSample[2]]
    print(f"here is the true label {true_label} and here is the predicted label:{class_predicted}")
    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if(true_label!=class_predicted):
        misclassifications+=1
#print the error rate
#--> add your Python code here
print(f"the error rate is: {misclassifications/len(db)}")






