# IMPLEMENTATION OF LINEAR REGRESSION USING GRADIENT DESCENT
## AIM:  
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## EQUIPMENT'S REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Import the required library and read the dataframe.  
2.Write a function computeCost to generate the cost function.   
3.Perform iterations of gradient steps with learning rate.  
4.Plot the Cost function using Gradient Descent and generate the required graph.   

## PROGRAM:
```
Program to implement the linear regression using gradient descent.
Developed by: RITHIGA SRI.B 
RegisterNumber: 212221230083
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit prediction")

def computeCost(X,y,theta):
    
    #Take in an numpy array X,y,theta and generate the cost function in a linear regression model
    
    m=len(y)#Length of training data
    h=X.dot(theta)#Hypothesis
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err)#returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)#Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
    #Take in numpy array X,y and theta and update theta by taking num_iters with learning rate of alpha
    #return theta and list of the cost of theta during each iteration
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*1/m*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
        
    return theta,J_history

theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

#Graph with Line of Best Fit
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="red")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

#Making predictions using optimized theta values
def predict(x,theta):
    #Takes in numpy array of x and theta and return the predicted value of y based on theta
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population= 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population= 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## OUTPUT:
* GRAPH PLOT FOR PROFIT PREDICTION:    
![image](https://user-images.githubusercontent.com/93427256/229015326-b253ed99-d2ff-4425-bc65-8f233a0eff4d.png)
* OUTPUT OF THE FUNCTION CALL:    
![image](https://user-images.githubusercontent.com/93427256/229016413-c6352355-e6ec-4097-bc7f-a30abfb26e94.png)
* GRADIENT DESCENT:    
![image](https://user-images.githubusercontent.com/93427256/229015651-acd855a6-4bc2-4ed9-a6f8-2999d1c88303.png)
* COST FUNCTION USING GRADIENT DESCENT:  
![image](https://user-images.githubusercontent.com/93427256/229015762-3a2d8945-a9cf-4fec-91a1-6954ddf4e21d.png)
* GRAPH WITH LINE OF BEST FIT:    
![image](https://user-images.githubusercontent.com/93427256/229015846-df92a4dd-ed5f-4d03-b771-96081fbca504.png)
### MAKING PREDICTIONS USING OPTIMIZED THETA VALUES:  
* (i)PROFIT PREDICTION FOR POPULATION OF 35000:    
![image](https://user-images.githubusercontent.com/93427256/229015991-514f5684-12c8-4c60-abbc-578e15200551.png)
* (ii)PROFIT PREDICTION FOR POPULATION OF 70000:    
![image](https://user-images.githubusercontent.com/93427256/229016071-8d1e6f8e-df0f-4d22-af09-c2fa739e8783.png)

## RESULT:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
