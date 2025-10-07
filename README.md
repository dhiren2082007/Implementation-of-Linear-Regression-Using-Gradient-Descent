# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. a# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Import Libraries: Import numpy, pandas, and StandardScaler from sklearn.preprocessing.

Read Data: Read '50_Startups.csv' into a DataFrame (data) using pd.read_csv().

Data Preparation:

Extract features (X) and target variable (y) from the DataFrame. Convert features to a numpy array (x1) and target variable to a numpy array (y). Scale the features using StandardScaler(). Linear Regression Function:

Define linear_regression(X1, y) function for linear regression. Add a column of ones to features for the intercept term. Initialize theta as a zero vector. Implement gradient descent to update theta. Model Training and Prediction:

Call linear_regression function with scaled features (x1_scaled) and target variable (y). Prepare new data for prediction by scaling and reshaping. Use the optimized theta to predict the output for new data. Print Prediction:

Inverse transform the scaled prediction to get the actual predicted value. Print the predicted value.
## Program:

```

/*
Program to implement the linear regression using gradient descent.
Developed by: Dhiren D
RegisterNumber: 25007814

import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
DATA.HEAD()

![Screenshot 2024-08-29 183415](https://github.com/user-attachments/assets/763022ca-616b-4141-942a-109ee33e0446)

X VALUE 

![Screenshot 2024-11-04 234336](https://github.com/user-attachments/assets/d83682e4-537f-4984-bfa1-06d7bcf56c0e)

X1_SCALED VALUE 

![Screenshot 2024-11-04 234422](https://github.com/user-attachments/assets/5a3585ab-7467-4dc2-a191-ac07075abb56)


PREDICTED VALUES:

![image](https://github.com/user-attachments/assets/eb585b7f-3b99-40c0-ab3a-9c6072a7ed39)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
```

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
