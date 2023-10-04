#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining=[]
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        #first row is the columns
        if(i>0):
            dbTraining.append (row)
num_map= {
    'Sunny':1, 'Overcast':2,'Rain':3,
    'Hot':1,'Mild':2,'Cool':3,
    'High':1,'Normal':2,
    'Weak':1,'Strong':2,
    'Yes':1,'No':2

}
#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X =[]
n=len(dbTraining[0])
for record in dbTraining:
    new_record=[]
    #dont include target
    for j in range (n-1):
        #skip over the record label/identifier
        if(j>0):
            new_record.append(num_map[record[j]])
    X.append(new_record)

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y =[]
for record in dbTraining:
    Y.append(num_map[record[n-1]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest=[]
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        #first row are the labels
        if(i>0):
            dbTest.append (row)

#printing the header os the solution
#--> add your Python code here
print("Probabilistic Predictions for Test Data:")
print("----------------------------------------")
#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
print("Day\t Outlook\t Temperature\t Humidity\t Wind\t PlayTennis\t Confidence\t")

n=len(dbTest[0])
for record in dbTest:
    new_record=[]
    #dont include target (it is missing and we need to make that prediction)
    for j in range (n-1):
        #skip over the identifier for the record
        if(j>0):
            new_record.append(num_map[record[j]])
    prediction=clf.predict_proba([new_record])[0]
    yes_confidence=round(prediction[0],4)
    no_confidence=round(prediction[1],4)
    if(yes_confidence>0.75 ):
        print(f"{record[0]}\t {record[1]}\t\t {record[2]}\t\t {record[3]}\t\t {record[4]}\t\t Yes \t\t{yes_confidence}\t")
    elif (no_confidence>0.75):
        print(f"{record[0]}\t {record[1]}\t {record[2]}\t\t {record[3]}\t\t {record[4]}\t\t No \t\t{no_confidence}\t")
