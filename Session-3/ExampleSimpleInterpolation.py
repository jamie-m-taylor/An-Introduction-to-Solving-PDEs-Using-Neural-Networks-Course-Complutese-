# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


##Produce some artificial, noisy data - training and validation
n_train = 100

x_train = tf.random.uniform([n_train,1])

y_train = tf.math.sin(4*np.pi*x_train)+tf.random.normal([n_train,1],stddev=0.05)

n_test = 20

x_test = tf.random.uniform([n_test,1])

y_test = tf.math.sin(4*np.pi*x_test)+tf.random.normal([n_test,1],stddev=0.05)



def build_model(neurons,activation):
    
    ##Define input shape
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input")
    
    
    ##A single hiddel layer
    l1 = tf.keras.layers.Dense(neurons, activation=activation)(xvals)

    ##Output
    out = tf.keras.layers.Dense(1)(l1)
    
    #Create the model and show information
    model = tf.keras.Model(inputs=xvals, outputs=out)
    model.summary()
    return model

class TrainingObjects:
    def __init__(self, neurons, activation="tanh"):
        
        #Create model
        self.u_model = build_model(neurons,activation)
        
        #Define optimiser
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
        
        #Various lists for saving information
        self.losslist = []
        self.vallist = []
        self.itslist = []
        
        
        #Import the data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    ##Define the loss function
    @tf.function
    def loss_fn(self,x,y):
        ##Simple mean squared error loss
        err = self.u_model(x)-y
        return tf.reduce_mean(err**2)
        
        
    ## Define the function that returns the loss and gradients when evaluated
    ## on the training set. 
    @tf.function
    def loss_grads(self,):
        with tf.GradientTape() as t1:
            l = self.loss_fn(self.x_train,self.y_train)
        return l,t1.gradient(l,self.u_model.trainable_weights)
    
    
    ##Take the gradient, apply it via the optimiser to update weights and 
    ## return the loss
    @tf.function
    def one_step(self,):
        l,g = self.loss_grads()
        self.optimizer.apply_gradients(zip(g,self.u_model.trainable_weights))
        return l
    
    
    ##Now define the training loop
    def train(self,iterations):
        self.i=0
        while self.i<iterations:
            self.i+=1
            
            ##This evaluates the loss, updates model via gradients. 
            l = self.one_step()
            
            ##Record the training and validation loss
            self.losslist+= [float(l)]
            val = self.loss_fn(self.x_test,self.y_test)
            self.vallist += [float(val)]
            self.itslist+=[self.i]
            
            ##Print information 
            print("Epoch:",self.i,"Loss:",float(l),"Val:",float(val))


iterations = 25000

neurons =50




history_list = []


size_list = []

T = TrainingObjects(neurons)
T.train(iterations)
loss_list = T.losslist
val_list = T.vallist


itlist = [i+1. for i in range(iterations)]


###Show the evolution of loss
plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
plt.plot(T.itslist[::100],T.losslist[::100], label="Training", linewidth=2)
plt.plot(T.itslist[::100],T.vallist[::100], label="Validation", linewidth=2)   # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")

plt.title("Loss evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Loss", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

xplot = tf.constant([i/100 for i in range(101)])


ue = tf.math.sin(4*np.pi*xplot)

u_final = T.u_model(xplot)


plt.plot(xplot, ue, label="Exact", linewidth=2, color="#2ca02c")
plt.plot(xplot, u_final, label="Final", linewidth=2, linestyle="--", color="#1f77b4")
plt.scatter(T.x_train, T.y_train, label="Data", s=20, alpha=1, color="black")

plt.xlabel("x", fontsize=13)
plt.ylabel("u(x)", fontsize=13)

plt.tick_params(labelsize=11)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.show()
