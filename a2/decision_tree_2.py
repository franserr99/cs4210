#-------------------------------------------------------------------------
# AUTHOR: Francisco Serrano
# FILENAME: decision_tree_2.py
# SPECIFICATION: Construct a decision tree using each of the training sets, compute the accuracy by predicting the test set and averaging over 10 runs for each.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
#read the test data and add this data to dbTest
    #--> add your Python code here
    #reading the training data in a csv file
dbTest=[]
with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append (row)

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)
    #transform the original categorical training features to numbers and add to the 4D array X
    num_map={
    'Young':1 , 'Myope':1,'Yes':1,'Reduced':1,
    'Prepresbyopic':2,'Hypermetrope':2, 'No':2,'Normal':2,
    'Presbyopic':3
    }  
    n=len(dbTraining[0])
    for record in dbTraining:
        new_record=[]
        #dont include target
        for i in range (n-1):
            new_record.append(num_map[record[i]])
        X.append(new_record)
    print("\n\n\nthe features:")
    print(X)
    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for record in dbTraining:
        Y.append(num_map[record[n-1]])
    print("target column/var:")
    print(Y)
    accuraries=[]
    #loop your training and test tasks 10 times here
    for i in range (10):
        true_positive=0
        true_negative=0
        false_positive=0
        false_negative=0

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        n=len(dbTest[0])
        for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
            matrix=[]
            new_record=[]
            #dont include target
            for i in range (n-1):
                new_record.append(num_map[data[i]])
            matrix.append(new_record)
            class_predicted = clf.predict(matrix)[0]
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            
            true_label=num_map[data[n-1]]
            #print(f"we predicted: {class_predicted} for {matrix} with a true label of {true_label}")
            #when i converted i mapped yes to 1 and no to 2
            if(class_predicted==true_label):
                if(class_predicted==1):
                    true_positive+=1
                elif(class_predicted==2):
                    true_negative+=1
            elif(class_predicted!=true_label ):
                #true label is a no, a 2, but we predicted as a yes so a false positive
                if(class_predicted==1):
                    false_positive+=1
                #predicted a no but it was a yes, so false neg
                elif(class_predicted==2):
                    false_negative+=1
        #print(f"true positive: {true_positive}")
        #print(f"true negative: {true_negative}")
        #print(f"false positive: {false_positive}")
        #print(f"false negative: {false_negative}")
        accuracy=(true_positive+true_negative)/(true_negative+true_positive+false_negative+false_positive)
        accuraries.append(accuracy)
    #find the average of this model during the 10 runs (training and test set)
    avg_accuracy=(sum(accuraries)/len(accuraries))
    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"final accuracy when training on {ds} was: {avg_accuracy}")