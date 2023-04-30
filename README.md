# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset.
2. Check for null and duplicate values.
3. Assign x and y values.
4. Split data into train and test data.
5. Import logistic regression and fit the training data.
6. Predict y value.
7. Calculate accuracy and confusion matrix.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Yugendaran.G
RegisterNumber: 212221220063 

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
print("Placement data:")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or coloumn
print("Salary data:")
data1.head()

print("Checking the null() function:")
data1.isnull().sum()

print ("Data Duplicate:")
data1.duplicated().sum()

print("Print data:")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

print("Data-status value of x:")
x=data1.iloc[:,:-1]
x

print("Data-status value of y:")
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print ("y_prediction array:")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#a library for large
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score 
accuracy=accuracy_score(y_test,y_pred) 
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
print("Confusion array:")
confusion

from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("Classification report:")
print(classification_report1)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
![image](https://user-images.githubusercontent.com/128135616/235350737-27e7237c-ceaa-4685-9169-49dd7ba915d0.png)

![image](https://user-images.githubusercontent.com/128135616/235350804-fe6113f5-386f-4a4e-b26c-21d651683564.png)

![image](https://user-images.githubusercontent.com/128135616/235352431-95f73d56-0387-478f-95cf-f1ef920ecb8e.png)

![image](https://user-images.githubusercontent.com/128135616/235352465-5854b6cb-1d08-4e90-9f6b-c09aad85f41f.png)

![image](https://user-images.githubusercontent.com/128135616/235352502-661f2d2a-64c4-455d-9c2b-6327263b564a.png)

![image](https://user-images.githubusercontent.com/128135616/235352550-facc1d8f-e413-44a9-89c6-06468f937ac9.png)

![image](https://user-images.githubusercontent.com/128135616/235352615-9fdec3b0-98a4-4740-95e0-d84af60e3e6f.png)

![image](https://user-images.githubusercontent.com/128135616/235352648-581c0a83-fc2d-4460-b918-cba3db9afcdd.png)

![image](https://user-images.githubusercontent.com/128135616/235352703-090bcdfe-220b-494e-805b-edc98a591359.png)

![image](https://user-images.githubusercontent.com/128135616/235352827-4911063c-5a2c-4c1a-8ceb-7b1b24c28b59.png)

![image](https://user-images.githubusercontent.com/128135616/235352893-a4d6572f-7189-47bc-b869-7acc522b12c1.png)

![image](https://user-images.githubusercontent.com/128135616/235352940-6bb927f0-8100-425d-80e3-351eaac46717.png)

![image](https://user-images.githubusercontent.com/128135616/235352965-e8bcbd0d-390a-44c3-bafd-1ee438c30f0a.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
