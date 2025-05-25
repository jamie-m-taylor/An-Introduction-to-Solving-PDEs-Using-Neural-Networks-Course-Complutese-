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
    def __init__(self, neurons,quad_rule,npts=50, activation="tanh"):
        self.neurons = neurons
        self.activation = activation
        self.u_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
        
        self.losslist = []
        self.errlist = []
        self.itslist = []
        
        tf.random.set_seed(123)
        
        self.n = npts
    
        self.quad_rule=quad_rule
        
        if quad_rule=="SMC":
            self.xmesh = tf.constant([i/self.n for i in range(self.n)])
        
        self.x_err = tf.constant([(i+0.5)/200 for i in range(200)])
    
    def u_exact(self,x):
        return tf.math.sin(np.pi*x*2)**3
    
    def gen_points(self,):
        if self.quad_rule =="SMC":
            x = self.xmesh + tf.random.uniform([self.n],maxval=1/self.n)
        else:
            x = tf.random.uniform([self.n])
        return x
    
    
    def _build_model(self):
        xvals = tf.keras.layers.Input(shape=(1,), name="x_input")
        
        
        l1 = tf.keras.layers.Dense(self.neurons, activation=self.activation)(xvals-0.5)

        out = tf.keras.layers.Dense(1)(l1)
        model = tf.keras.Model(inputs=xvals, outputs=out)
        model.summary()
        return model
    
    def err(self,):
        return tf.reduce_mean((tf.squeeze(self.u_model(self.x_err))-self.u_exact(self.x_err))**2)**0.5
    
    def loss_fn(self,):
        x = self.gen_points()
        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                uerr = tf.squeeze(self.u_model(x))-self.u_exact(x)
            duerr = t1.gradient(uerr,x)
        dduerr = t2.gradient(duerr,x)
        
        u0=self.u_model(tf.constant([0.,1.]))
                
        return tf.reduce_mean((duerr)**2)+tf.reduce_sum(u0**2)
        
    
    @tf.function
    def loss_grads(self,):
        with tf.GradientTape() as t1:
            l = self.loss_fn()
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
            self.errlist+=[self.err()]
            self.i+=1
            print("Epoch:",self.i,"Loss:",float(l),"Err:",float(self.errlist[-1]))


iterations = 20000

neurons =50

loss_list = []
err_list = []
ufinal = []

quad_rules = ["MC","SMC"]

for quad_rule in quad_rules:
    T = TrainingObjects(neurons,quad_rule)
    T.train(iterations)
    loss_list += [T.losslist]
    err_list += [T.errlist]
    ufinal +=[T.u_model(T.x_err)]
    


itlist = [i+1. for i in range(iterations)]



plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
plt.plot(itlist[::100],loss_list[0][::100], label="MC", linewidth=2)
plt.plot(itlist[::100],loss_list[1][::100], label="SMC", linewidth=2)   # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")

plt.title("Loss evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Loss", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()


plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
plt.plot(itlist[::100],err_list[0][::100], label="MC", linewidth=2)
plt.plot(itlist[::100],err_list[1][::100], label="SMC", linewidth=2)   # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")

plt.title("Error evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Error (L^2)", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()


plt.figure(figsize=(4, 3), dpi=200)

xplot = T.x_err

plt.plot(xplot, T.u_exact(xplot), label="Exact", linewidth=2, color="#2ca02c")
plt.plot(xplot, ufinal[0], label="MC", linewidth=2, linestyle="--", color="#1f77b4")
plt.plot(xplot, ufinal[1], label="SMC", linewidth=2, linestyle=":", color="#d62728")

plt.xlabel("x", fontsize=13)
plt.ylabel("u(x)", fontsize=13)

plt.tick_params(labelsize=11)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.show()
