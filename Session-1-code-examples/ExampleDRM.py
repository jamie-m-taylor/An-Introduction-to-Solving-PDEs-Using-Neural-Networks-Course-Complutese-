# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os



class TrainingObjects:
    def __init__(self, neurons, activation="tanh"):
        self.neurons = neurons
        self.activation = activation
        self.u_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
        
        self.losslist = []
        self.vallist = []
        self.itslist = []
        
        tf.random.set_seed(123)
        
        self.x_test = tf.random.uniform([15])
        
        self.x_train = tf.random.uniform([15])
        
        
        
        self.loss_best=10**10
        
        self.best_weights=[]
        
    
    def u_exact(self,x):
        return tf.nn.relu(x-0.5)
    
    
    
    def _build_model(self):
        xvals = tf.keras.layers.Input(shape=(1,), name="x_input")
        
        
        l1 = tf.keras.layers.Dense(self.neurons, activation=self.activation)(xvals-0.5)

        out = tf.keras.layers.Dense(1)(l1)
        model = tf.keras.Model(inputs=xvals, outputs=out)
        model.summary()
        return model
    
    def loss_fn(self,x):
        
        with tf.GradientTape() as t1:
            t1.watch(x)
            uerr = tf.squeeze(self.u_model(x))-self.u_exact(x)
        duerr = t1.gradient(uerr,x)
        
        u0=self.u_model(tf.constant([0.]))
                
        return tf.reduce_mean((duerr)**2)+tf.reduce_sum(u0**2)
        
    
    @tf.function
    def loss_grads(self,):
        with tf.GradientTape() as t1:
            l = self.loss_fn(self.x_train)
        return l,t1.gradient(l,self.u_model.trainable_weights)
    
    @tf.function
    def one_step(self,):
        l,g = self.loss_grads()
        self.optimizer.apply_gradients(zip(g,self.u_model.trainable_weights))
        return l
    
    def train(self,iterations):
        self.i=0
        while self.i<iterations:
            l = self.one_step()
            self.losslist+= [l]
            val = self.loss_fn(self.x_test)
            
            self.vallist+=[float(val)]
            if val<self.loss_best:
                self.loss_best = val
                self.best_weights = self.u_model.get_weights()
            self.i+=1
            print("Epoch:",self.i,"Loss:",float(l),"Val:",float(val))
        self.final_weights = self.u_model.get_weights()


iterations = 20000

neurons =50




history_list = []

h2list = []



size_list = []

T = TrainingObjects(neurons)
T.train(iterations)
loss_list = T.losslist
val_list = T.vallist


itlist = [i+1. for i in range(iterations)]



plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
plt.plot(itlist[::100],loss_list[::100], label="Training", linewidth=2)
plt.plot(itlist[::100],val_list[::100], label="Validation", linewidth=2)   # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")

plt.title("Loss evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Loss", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

xplot = tf.constant([i/100 for i in range(101)])


with tf.GradientTape(persistent=True) as t1:
    t1.watch(xplot)
    ue = T.u_exact(xplot)

    u_final = T.u_model(xplot)

due = t1.gradient(ue,xplot)
du_final = t1.gradient(u_final,xplot)

with tf.GradientTape() as t1:
    t1.watch(T.x_train)
    ud = T.u_exact(T.x_train)
dud = t1.gradient(ud,T.x_train)

T.u_model.set_weights(T.best_weights)


with tf.GradientTape(persistent=True) as t1:
    t1.watch(xplot)
    u_best = T.u_model(xplot)
du_best = t1.gradient(u_best,xplot)

plt.figure(figsize=(4, 3), dpi=200)

plt.plot(xplot, ue, label="Exact", linewidth=2, color="#2ca02c")
plt.plot(xplot, u_final, label="Final", linewidth=2, linestyle="--", color="#1f77b4")
plt.plot(xplot, u_best, label="Best", linewidth=2, linestyle=":", color="#d62728")

plt.scatter(T.x_train, ud, label="Data", s=20, alpha=1, color="black")

plt.xlabel("x", fontsize=13)
plt.ylabel("u(x)", fontsize=13)

plt.tick_params(labelsize=11)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 3), dpi=200)

plt.plot(xplot, due, label="Exact", linewidth=2, color="#2ca02c")
plt.plot(xplot, du_final, label="Final", linewidth=2, linestyle="--", color="#1f77b4")
plt.plot(xplot, du_best, label="Best", linewidth=2, linestyle=":", color="#d62728")

plt.scatter(T.x_train, dud, label="Data", s=20, alpha=1, color="black")

plt.xlabel("x", fontsize=13)
plt.ylabel("u'(x)", fontsize=13)

plt.tick_params(labelsize=11)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.show()


