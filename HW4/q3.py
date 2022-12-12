#!/usr/bin/env python
# coding: utf-8

# **NAME - PRAKHAR BHARDWAJ**
# 
# **ANDREW ID - prakharb**
# 
# **Q3** 

# ### Note for question3
# - Please follow the template to complete q3
# - You may create new cells to report your results and observations

# In[6]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## P1. Load data and plot
# ### TODO
# - load q3_data.csv
# - plot the points of different labels with different color

# In[7]:


# Load dataset
data=pd.read_csv("q3_data.csv",header=None)
x1=data.iloc[:,0]
x2=data.iloc[:,1]
labels=data.iloc[:,2]

print(x1.shape)
# Plot points
plt.scatter(x1,x2,c=labels)
plt.show()


# ## P2. Feature mapping
# ### TODO
# - implement function **map_feature()** to transform data from original space to the 28D space specified in the write-up

# In[8]:


# Transform points to 28D space
def map_feature(x1,x2):
    dimension=28
    deg=6
    mf=np.ones((len(x1),dimension))
    col=0
    for i in range(1,deg+1):
        for j in range(i+1):
            col+= 1
            mf[:,col]=x1**(i-j)*x2**j
    return mf
inputs=map_feature(x1,x2).T
print(inputs.shape)


# ## P3. Regularized Logistic Regression
# ### TODO
# - implement function **logistic_regpression_regularized()** as required in the write-up
# - draw the decision boundary
# 
# ### Hints
# - recycling code from HW2 is allowed
# - you may use functions defined this section for part 4 below
# - although optional for the report, plotting the convergence curve will be helpful

# In[9]:


# Define your functions here
def sigmoid(value):
    sig=1/(1+np.exp(-value))
    return sig

def gradb0(inputs,inBrkt,labels):
    return -((inBrkt @ inputs.T)/len(labels))

def gradb(inputs,inBrkt,labels,lamb,weights):
    return (-(inBrkt @ inputs.T)/len(labels)+(lamb/len(labels))*weights[1:])

def cost(predicted,labels,weights,lamb):
    term=((-labels*np.log(predicted))-((1-labels)*np.log(1-predicted)))
    costs=(np.sum(term)/len(labels))+lamb*np.sum(weights[1:,]**2)/(2*len(labels))
    return cost

def logistic_regression_regularized(inputs,weights,labels,number_steps,learning_rate,lamb):
    checkCost=[]
    
    for i in range(number_steps):
        bTx=np.dot(weights.T,inputs)
        predicted=sigmoid(bTx)
        inBrkt=labels-predicted
        grad0=gradb0(inputs[0],inBrkt,labels)
        gradRest=gradb(inputs[1:],inBrkt,labels,lamb,weights)
        weights[0]=weights[0]-learning_rate*grad0
        weights[1:]=weights[1:]-learning_rate*gradRest
    return weights,checkCost

weights=np.zeros(28)
number_steps=10000
learning_rate=0.01
labels=np.array(labels)
lamb=1

weights,cost=logistic_regression_regularized(inputs,weights,labels,number_steps,learning_rate,lamb)
# print(weights)

predt=sigmoid(np.dot(weights,inputs))
predt[np.where(predt>0.5)]=1
predt[np.where(predt<0.5)]=0
acc=0
for i in range(len(predt)):
    if predt[i]==labels[i]:
        acc+=1
print("Accuracy when lambda is 1:",acc/len(predt))


# Plot decision boundary
x=np.linspace(np.amin(x1),np.amax(x1),100)
y=np.linspace(np.amin(x2),np.amax(x2),100)
j1,j2=np.meshgrid(x,y)
gridX=j1.ravel()
gridY=j2.ravel()
g1=np.array(gridX).T
g2=np.array(gridY).T
pltGrid=map_feature(g1,g2)
sig=sigmoid(np.dot(weights,pltGrid.T))
plt.scatter(x1[labels==0],x2[labels==0], color = "red")
plt.scatter(x1[labels==1],x2[labels==1], color = "black")
plt.contour(j1,j2,sig.reshape(100,100),[0.5])


# ## P4. Tune the strength of regularization
# ### TODO
# - tweak the hyper-parameter $\lambda$ to be $[0, 1, 100, 10000]$
# - draw the decision boundaries
# 

# In[10]:


# lambda = 0
weights=np.zeros(28)
number_steps=10000
learning_rate=0.01
labels=np.array(labels)
lamb=0
weights,cost=logistic_regression_regularized(inputs,weights,labels,number_steps,learning_rate,lamb)
# print(weights)
predt=sigmoid(np.dot(weights,inputs))
predt[np.where(predt>0.5)]=1
predt[np.where(predt<0.5)]=0
acc=0
for i in range(len(predt)):
    if predt[i]==labels[i]:
        acc+=1
print("Accuracy when lambda is 0:",acc/len(predt))

# Plot decision boundary
x=np.linspace(np.amin(x1),np.amax(x1),100)
y=np.linspace(np.amin(x2),np.amax(x2),100)
j1,j2=np.meshgrid(x,y)
gridX=j1.ravel()
gridY=j2.ravel()
g1=np.array(gridX).T
g2=np.array(gridY).T
pltGrid=map_feature(g1,g2)
sig=sigmoid(np.dot(weights,pltGrid.T))
plt.scatter(x1[labels==0],x2[labels==0], color = "red")
plt.scatter(x1[labels==1],x2[labels==1], color = "black")
plt.contour(j1,j2,sig.reshape(100,100),[0.5])
plt.show()


# lambda = 1
weights=np.zeros(28)
number_steps=10000
learning_rate=0.01
labels=np.array(labels)
lamb=1
weights,cost=logistic_regression_regularized(inputs,weights,labels,number_steps,learning_rate,lamb)
# print(weights)
predt=sigmoid(np.dot(weights,inputs))
predt[np.where(predt>0.5)]=1
predt[np.where(predt<0.5)]=0
acc=0
for i in range(len(predt)):
    if predt[i]==labels[i]:
        acc+=1
print("Accuracy when lambda is 1:",acc/len(predt))

# Plot decision boundary
x=np.linspace(np.amin(x1),np.amax(x1),100)
y=np.linspace(np.amin(x2),np.amax(x2),100)
j1,j2=np.meshgrid(x,y)
gridX=j1.ravel()
gridY=j2.ravel()
g1=np.array(gridX).T
g2=np.array(gridY).T
pltGrid=map_feature(g1,g2)
sig=sigmoid(np.dot(weights,pltGrid.T))
plt.scatter(x1[labels==0],x2[labels==0], color = "red")
plt.scatter(x1[labels==1],x2[labels==1], color = "black")
plt.contour(j1,j2,sig.reshape(100,100),[0.5])
plt.show()


# lambda = 100
weights=np.zeros(28)
number_steps=10000
learning_rate=0.01
labels=np.array(labels)
lamb=100
weights,cost=logistic_regression_regularized(inputs,weights,labels,number_steps,learning_rate,lamb)
# print(weights)
predt=sigmoid(np.dot(weights,inputs))
predt[np.where(predt>0.5)]=1
predt[np.where(predt<0.5)]=0
acc=0
for i in range(len(predt)):
    if predt[i]==labels[i]:
        acc+=1
print("Accuracy when lambda is 100:",acc/len(predt))

# Plot decision boundary
x=np.linspace(np.amin(x1),np.amax(x1),100)
y=np.linspace(np.amin(x2),np.amax(x2),100)
j1,j2=np.meshgrid(x,y)
gridX=j1.ravel()
gridY=j2.ravel()
g1=np.array(gridX).T
g2=np.array(gridY).T
pltGrid=map_feature(g1,g2)
sig=sigmoid(np.dot(weights,pltGrid.T))
plt.scatter(x1[labels==0],x2[labels==0], color="red")
plt.scatter(x1[labels==1],x2[labels==1], color = "black")
plt.contour(j1,j2,sig.reshape(100,100),[0.5])
plt.show()


# lambda = 10000
weights=np.zeros(28)
number_steps=10000
learning_rate=0.01
labels=np.array(labels)
lamb=10000
weights,cost=logistic_regression_regularized(inputs,weights,labels,number_steps,learning_rate,lamb)
# print(weights)
predt=sigmoid(np.dot(weights,inputs))
predt[np.where(predt>0.5)]=1
predt[np.where(predt<0.5)]=0
acc=0
for i in range(len(predt)):
    if predt[i]==labels[i]:
        acc+=1
print("Accuracy when lambda is 10000:",acc/len(predt))

# Plot decision boundary
x=np.linspace(np.amin(x1),np.amax(x1),100)
y=np.linspace(np.amin(x2),np.amax(x2),100)
j1,j2=np.meshgrid(x,y)
gridX=j1.ravel()
gridY=j2.ravel()
g1=np.array(gridX).T
g2=np.array(gridY).T
pltGrid=map_feature(g1,g2)
sig=sigmoid(np.dot(weights,pltGrid.T))
plt.scatter(x1[labels==0],x2[labels==0], color = "red")
plt.scatter(x1[labels==1],x2[labels==1], color = "black")
plt.contour(j1,j2,sig.reshape(100,100),[0.5])
plt.show()


# ## d)
# 
# 

# Lamba = 0, 1, 100, and 10,000 are compared: The regularization grows and we become more generalized as the lambda value increases (less "overfitted" decision boundary). As a result, the accuracy slightly declines, but the decision boundary is more broadly defined. There is no regularization of the data set using lambda 0. Among the values of lambda 0, 1, 100, and 1000, 100 is the most ideal. Additionally, it is evident that when lambda increases, weight values decrease, which causes the impact of some features in the prediction hypothesis to decrease.

# In[ ]:




