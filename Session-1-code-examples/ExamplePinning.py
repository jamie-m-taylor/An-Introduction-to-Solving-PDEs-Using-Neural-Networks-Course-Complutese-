# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class bc_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(bc_layer,self).__init__()
        
    def call(self,inputs):
        
        xy,pu = inputs
        r = tf.reduce_sum(xy**2,axis=-1)
        cut = 1-r
        return tf.einsum("i,ij->i",cut,pu)

class TrainingObjects:
    def __init__(self, neurons,pinning, activation=tf.math.tanh):
        self.neurons = neurons
        self.activation = activation
        self.pinning = pinning
        
        self.u_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
        
        self.losslist = []
        self.l2list = []
        self.h2list = []
        self.itslist = []
        
        tf.random.set_seed(123)
        
        self.n1 = 1000
        self.n2 = 200
        
        
        
        self.w = 100.
        
        rtest = tf.constant([(i+0.5)/50 for i in range(10)])
        ttest = tf.constant([(i+0.5)*2*np.pi/100 for i in range(50)])
        
        R,T = tf.meshgrid(rtest,ttest)
        R=tf.reshape(R,[50*10])
        T=tf.reshape(T,[50*10])
        
        self.xtest = tf.math.cos(T)*R
        self.ytest = tf.math.sin(T)*R
        self.Rtest = R
        
    def u_exact(self, x,y):
        p2 = tf.math.sin(1-x**2-y**2)**3
        return p2
    
    
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
    
    def _build_model(self):
        xvals = tf.keras.layers.Input(shape=(2,), name="x_input")
        
        
        l1 = tf.keras.layers.Dense(self.neurons, activation=self.activation,
                                   kernel_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1))(xvals)
        l2 = tf.keras.layers.Dense(self.neurons,activation=self.activation)(l1)

        if self.pinning:
            p_out = tf.keras.layers.Dense(1)(l2)
            out = bc_layer()([xvals,p_out])
        else:
            out = tf.keras.layers.Dense(1)(l2)
        model = tf.keras.Model(inputs=xvals, outputs=out)
        model.summary()
        return model
    
    
    @tf.function
    def err(self,):
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.xtest)
            t1.watch(self.ytest)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(self.xtest)
                t2.watch(self.ytest)
                xy = tf.stack([self.xtest,self.ytest],axis=-1)
                err = tf.squeeze(self.u_model(xy))-self.u_exact(self.xtest,self.ytest)
            derrx,derry = t2.gradient(err,[self.xtest,self.ytest])
        h2part = t1.gradient(derrx,self.xtest)**2+2*t1.gradient(derry,self.xtest)**2+t1.gradient(derry,self.ytest)**2
        
        errh2 = tf.reduce_mean(self.Rtest*(h2part+err**2))**0.5
        errl2 = tf.reduce_mean(self.Rtest*err**2)**0.5
        return errh2,errl2
    
    def loss_fn(self,):
        x,y = self.sample_disc(self.n1)
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(y)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                xy = tf.stack([x,y],axis=-1)
                u = tf.squeeze(self.u_model(xy))-self.u_exact(x,y)
            dux,duy = t2.gradient(u,[x,y])
        lapu = t1.gradient(dux,x)+t1.gradient(duy,y)
        loss_pde = tf.reduce_mean((lapu)**2)
        
        
        if self.pinning:
            loss = loss_pde
            
        else:
            x0,y0 = self.sample_circle(self.n2)
            xy0 = tf.stack([x0,y0],axis=-1)
            u0 = tf.squeeze(self.u_model(xy0))-self.u_exact(x0,y0)
            
            
            
            loss_bc = tf.reduce_mean(u0**2)
            
            loss = loss_pde+self.w*loss_bc
        return loss
    
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
            errh2, errl2 = self.err()
            self.l2list+=[errl2]
            self.h2list += [errh2]
            self.i+=1
            self.itslist +=[self.i]
            print("Epoch:",self.i,"Loss:",float(l),"L^2:",float(errl2),"H^2",float(errh2))


iterations = 20000

neurons =25

loss_list = []
l2_list = []
h2_list = []
ufinal = []

pinning_list = [True,False]

for pinning in pinning_list:
    T = TrainingObjects(neurons,pinning)
    T.train(iterations)
    loss_list += [T.losslist]
    l2_list += [T.l2list]
    h2_list += [T.h2list]
    

final_l2 = [np.mean(l2[-100:-1]) for l2 in l2_list]

final_h2 = [np.mean(h2[-100:-1]) for h2 in h2_list]


labels = ["Pinning",'Penalty']



plt.figure(figsize=(4,3),dpi=250) 
for i in range(2):
    plt.plot(T.itslist,h2_list[i],label=labels[i])
plt.xscale("log")
plt.yscale('log')
plt.title("H2 error")
plt.xlabel("Iterations")
plt.ylabel("H2 error")
plt.legend()
plt.show()



plt.figure(figsize=(4,3),dpi=250) 
for i in range(2):
    plt.plot(T.itslist,l2_list[i],label=labels[i])
plt.xscale("log")
plt.yscale('log')
plt.title("L2 error")
plt.xlabel("Iterations")
plt.ylabel("L2 error")
plt.legend()
plt.show()

plt.figure(figsize=(4,3),dpi=250) 
for i in range(2):
    plt.plot(T.itslist,loss_list[i],label=labels[i])
plt.xscale("log")
plt.yscale('log')
plt.title("Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
