# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the python pandas library as pd
3. Read the contents of the Spam csv file
4. Display the first 5 rows of the dataset using head()
5. Assign x as v1 values and y as v2 values
6. From sklearn library select the feature extraction and import CountVectorizer
7. CountVectorizer will convert the Text to Numerical Data
8. From sklearn library import Support Vector Classifier (ie. SVC)
9. Predict the x_test using SVC
10. Print the accuracy of the SVM Model 11.Stop the program 

## Program:
```python
Program to implement the SVM For Spam Mail Detection..
Developed by: KULASEKARAPANDIAN K
RegisterNumber: 212222240052
```

```python
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding = 'Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

#### RESULT:
![Alt text](279721551-531072fd-1b41-4f23-a772-1c49449463b7.png)


#### HEAD VALUES:
![Alt text](279721680-8371228c-d7b6-4d48-990c-8272333a2639.png)

#### DATA INFO:
![Alt text](279721839-7b2f7cbb-95de-4321-98c1-b7e1d606f9fe.png)

#### NULL:
![Alt text](279721964-e3b7dbc0-975c-4a74-90e0-c732a55303e5.png)

#### PREDICTION VALUE:
![Alt text](279722154-e7928039-8e22-4a76-908c-29d5389b9916.png)

#### ACCURACY VALUE:
![Alt text](279722312-ac3d359e-8bf8-4104-9c5e-c941bb36f8a1.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
