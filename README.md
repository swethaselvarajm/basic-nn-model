# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Developing a neural network regression model involves designing a network with an input layer, one or more hidden layers, and an output layer for continuous values. The model is trained using a dataset where input features correspond to known target values. Key steps include selecting the architecture, such as the number of neurons and activation functions, and defining the loss function (e.g., mean squared error) to minimize prediction errors. The model's weights are optimized through backpropagation and gradient descent. Regularization techniques like dropout or L2 can prevent overfitting, ensuring the model generalizes well to new data.


## Neural Network Model

![image](https://github.com/user-attachments/assets/deb09092-f829-443a-81ab-588cd300cce8)


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
### Name: SWETHA S
### Register Number: 212222230155

## Importing Required packages
```
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

```
## Authenticate the Google sheet
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Data').sheet1

```
## Split the testing and training data
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=40)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)

```

## Build the Deep learning Model
```
ai_brain=Sequential([
    Dense(9,activation = 'relu',input_shape=[1]),
    Dense(16,activation = 'relu'),
     Dense(1)
])
ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train.astype(np.float32),epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
## Evaluate the Model
```
test=Scaler.transform(x_test)
ai_brain.evaluate(test,y_test.astype(np.float32))
n1=[[40]]
n1_1=Scaler.transform(n1)
ai_brain.predict(n1_1)
```

## Dataset Information

![dataset](https://github.com/user-attachments/assets/2fb4bccd-4bee-4aca-831f-cc7e91c3d867)



## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/6c620c25-408f-4696-8c00-24f6363b52df)

![image](https://github.com/user-attachments/assets/f2b9b010-d7b5-4efc-b2dc-d785639cc362)

### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/c4dbdabf-5733-45ca-9c89-ad344f5ac150)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/c4872d82-584e-44ff-a836-1f314a32be3f)


## RESULT
Thus a Neural network for Regression model is implemented.

