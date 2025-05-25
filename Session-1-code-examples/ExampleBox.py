# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def rhs(x,y):
    return -32*tf.math.sin(4*x)*tf.math.cos(4*y)


def u_exact(x,y):
    return tf.math.sin(4*x)*tf.math.cos(4*y)



class TrainingObjects:
    def __init__(self, neurons, box, activation="tanh"):
        self.neurons = neurons
        self.activation = activation
        self.box = box
        self.u_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3.5)
        
        self.losslist = []
        self.errlist = []
        self.itslist = []
                
        self.n1 = 1000
        self.n2 = 100
        
        
        rtest = tf.constant([(i+0.5)/50 for i in range(10)])
        ttest = tf.constant([(i+0.5)*2*np.pi/100 for i in range(50)])
        
        R,T = tf.meshgrid(rtest,ttest)
        R=tf.reshape(R,[50*10])
        T=tf.reshape(T,[50*10])
        
        self.xtest = tf.math.cos(T)*R
        self.ytest = tf.math.sin(T)*R
        self.Rtest = R
        
        
    def make_box_vals(self,neurons):
        t = tf.random.uniform([neurons],maxval=np.pi)
        n = tf.stack([tf.math.cos(t),tf.math.sin(t)],axis=-1)
        t1= tf.random.uniform([neurons],maxval=np.pi)
        r1 = tf.random.uniform([neurons])**0.5
        p = tf.einsum("i,ij->ij",r1,tf.stack([tf.math.cos(t1),tf.math.cos(t1)],axis=-1))
        k = tf.random.uniform([neurons],minval=-6/neurons**0.5,maxval=6/neurons**0.5)
        A = tf.einsum("i,ij->ji",k,n)
        b = tf.einsum("ij,ij,i->i",-p,n,k)
        return A,b
    
    def reinit_u(self,u_model,n):
        A,b=self.make_box_vals(n)
        u_model.layers[1].weights[0].assign(A)
        u_model.layers[1].weights[1].assign(b)
        
        
        
    
    def _build_model(self):
        xvals = tf.keras.layers.Input(shape=(2,), name="x_input")
        
        l1 = tf.keras.layers.Dense(self.neurons, activation=self.activation)(xvals)

        out = tf.keras.layers.Dense(1)(l1)
        model = tf.keras.Model(inputs=xvals, outputs=out)
        if self.box:
            self.reinit_u(model,self.neurons)
        model.summary()
        return model
    
        
    def sample_disc(self,n):
        r=tf.random.uniform([n])**0.5
        t = tf.random.uniform([n],maxval=2*np.pi)
        x = r*tf.math.cos(t)
        y = r*tf.math.sin(t)
        return x,y
    
    def sample_circle(self,n):
        t = tf.random.uniform([n],maxval=2*np.pi)
        x = tf.math.cos(t)
        y = tf.math.sin(t)
        return x,y    
    def loss_fn(self,):
        x,y = self.sample_disc(self.n1)
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(y)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                xy = tf.stack([x,y],axis=-1)
                u = self.u_model(xy)
            dux,duy = t2.gradient(u,[x,y])
        lapu = t1.gradient(dux,x)+t1.gradient(duy,y)
        
        
        
        x0,y0 = self.sample_circle(self.n2)
        xy0 = tf.stack([x0,y0],axis=-1)
        u0 = tf.squeeze(self.u_model(xy0))-u_exact(x0,y0)
        
        loss_ode = tf.reduce_mean((lapu-rhs(x,y))**2)
        
        loss_bc = tf.reduce_mean(u0**2)
        return loss_ode+10*loss_bc
    
    @tf.function
    def loss_grads(self,):
        with tf.GradientTape() as t1:
            l = self.loss_fn()
        return l,t1.gradient(l,self.u_model.trainable_weights)
    
    @tf.function
    def h2_error(self,):
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.xtest)
            t1.watch(self.ytest)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(self.xtest)
                t2.watch(self.ytest)
                xy = tf.stack([self.xtest,self.ytest],axis=-1)
                err = tf.squeeze(self.u_model(xy))-u_exact(self.xtest,self.ytest)
            derrx,derry = t2.gradient(err,[self.xtest,self.ytest])
        h2part = t1.gradient(derrx,self.xtest)**2+2*t1.gradient(derry,self.xtest)**2+t1.gradient(derry,self.ytest)**2
        
        errh1 = tf.reduce_mean(self.Rtest*(h2part+err**2))**0.5
        return errh1
    
    @tf.function
    def one_step(self,):
        l,g = self.loss_grads()
        self.optimizer.apply_gradients(zip(g,self.u_model.trainable_weights))
        return l
    
    def train(self,iterations):
        self.i=0
        while self.i<iterations:
            l = self.one_step()
            self.errlist+= [self.h2_error()]
            self.losslist+= [l]
            self.i+=1
            print("Epoch:",self.i,"Loss:",float(l),"Error:",float(self.errlist[-1]))


box_list = [True,False]

final_errors = []
iterations = 25000

neurons =50





history_list = []

h2list = []



size_list = []

for box in box_list:
    T = TrainingObjects(neurons, box)
    T.train(iterations)
    history_list +=[T.losslist]
    h2list +=[T.errlist]


itlist = [i+1. for i in range(iterations)]

labels = ["Box","Standard"]

plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
for i in range(len(history_list)):
    history = history_list[i]
    plt.plot(history, label=labels[i], linewidth=2)  # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Loss", fontsize=12)  # Label for y-axis
plt.legend(title="Offset", loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

# Plot error over iterations
plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
for i in range(len(history_list)):
    history = h2list[i]
    plt.plot(history, label=labels[i], linewidth=2)  # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Error", fontsize=12)  # Label for y-axis
plt.legend(title="Offset", loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()