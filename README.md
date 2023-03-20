# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

## Neural Network Model

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://user-images.githubusercontent.com/93427246/224904957-962c297b-72c3-43f9-b361-22e9b7efb8b9.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed By: Shyam Kumar A
Register Number: 212221230098

from google.colab import auth

import gspread

from google.auth import default

import pandas as pd

auth.authenticate_user()

creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DL').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df1 = df.astype({'ip':'int','op':'int'})
df1.head()

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df1[["ip"]].values
y = df1[["op"]].values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.33,random_state=33)

Scaler = MinMaxScaler()

Scaler.fit(xtrain)

xtrain1 = Scaler.transform(xtrain)

AiBrain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])

AiBrain.compile(optimizer = 'rmsprop',loss = 'mse')

AiBrain.fit(xtrain1,ytrain,epochs = 2000)

loss_df = pd.DataFrame(AiBrain.history.history)

loss_df.plot()

xtest1 = Scaler.transform(xtest)

AiBrain.evaluate(xtest1,ytest)

xn1 = [[10]]

xn1_1 = Scaler.transform(xn1)

AiBrain.predict(xn1_1)mport default

```

## Dataset Information
![image](https://user-images.githubusercontent.com/93427246/224903108-600c4ce3-42ef-44bd-b0e5-770bbc90a9c7.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/93427246/224903390-501f127b-1536-4844-afe3-1dff611dae17.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/93427246/224903932-7e06c013-2f5a-4d10-97af-6a18c71b3072.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/93427246/224904405-6ec36239-184e-4c41-9d30-7e9010818ffa.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
