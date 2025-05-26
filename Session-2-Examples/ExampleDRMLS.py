# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
tf.keras.backend.set_floatx('float64')
sin = tf.math.sin
cos = tf.math.cos
Pi=np.pi      

class bc_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(bc_layer,self).__init__()
        
    def call(self,inputs):
        
        x,pu = inputs
        
        cut =tf.reshape(x*(1-x),[-1])
        
        return tf.einsum("i,ij->ij",cut,pu)


class TrainingObjects:
    def __init__(self, neurons,npts, activation=tf.math.tanh,LS=False):
        self.neurons = neurons
        self.activation = activation
        self.u_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
        
        
        self.npts = npts
        self.xmesh=tf.constant([i/self.npts for i in range(self.npts)],dtype="float64")
        
        self.losslist = []
        self.h1list = []
        self.itslist = []
        
        self.LS=LS
        if not LS:
            self.c=tf.Variable(0.1*tf.random.uniform([10],minval=-1,dtype="float64"))
        
    def rhs(self,x):
        return 32*Pi*cos(16*Pi*x) - 256*x*Pi**2*sin(16*Pi*x)
    
    def u_exact(self,x):
        return x*tf.math.sin(16*np.pi*x)
    
    
    def _build_model(self):
        xvals = tf.keras.layers.Input(shape=(1,), name="x_input")
        
        
        l1 = tf.keras.layers.Dense(self.neurons, activation=self.activation,
                                   kernel_initializer = tf.keras.initializers.RandomUniform(minval=-3*np.pi,maxval=3*np.pi),
                                   bias_initializer = tf.keras.initializers.RandomUniform(minval=-3*np.pi,maxval=3*np.pi)
                                   )(xvals-0.5)

        p_out = tf.keras.layers.Dense(10,activation=self.activation)(l1)
        out = bc_layer()([xvals,p_out])
        model = tf.keras.Model(inputs=xvals, outputs=out)
        model.summary()
        return model
    
    
    @tf.function
    def gen_points(self,)    :
        U = tf.random.uniform([self.npts],maxval=1/self.npts,dtype="float64")
        return tf.concat([self.xmesh+U,self.xmesh+1/self.npts-U],axis=-1)
    
    @tf.function
    def h1_error(self,):
        x = tf.constant([(i+0.5)/200 for i in range(200)],dtype="float64")
        with tf.GradientTape() as t1:
            t1.watch(x)
            if self.LS:
                c=self.LS_system()
            else:
                c=self.c
            ub = self.u_model(x)
            uerr = tf.einsum("xi,i->x",ub,c)-self.u_exact(x)
        duerr = t1.gradient(uerr,x)
        return tf.reduce_mean(duerr**2)**0.5

    
    @tf.function
    def LS_system(self,):
        x1 = self.gen_points()
        with tf.autodiff.ForwardAccumulator(primals=x1, tangents=tf.ones_like(x1)) as acc:
            u = self.u_model(x1)
        du = acc.jvp(u)
        
        G = tf.einsum("xi,xj->ij",du,du)/(2*self.npts)
        
        
        
        x1 = self.gen_points()
        
        
        u = self.u_model(x1)
        f = self.rhs(x1)
        
        F =tf.reshape(tf.einsum("xi,x->i",u,f),[-1,1])/(2*self.npts)
        
        c= -tf.reshape(tf.linalg.solve(G,F),[-1])
        
        return c
    
    @tf.function
    def loss_fn_LS(self,c):
        
        x = self.gen_points()
        with tf.GradientTape() as t1:
            t1.watch(x)
            ub = self.u_model(x)
            u = tf.einsum("xi,i->x",ub,c)
            
        du = t1.gradient(u,x)
        f = self.rhs(x)
        
        integrand = du**2/2+f*u
        
                
        return tf.reduce_mean(integrand)
        
    
    @tf.function
    def loss_grads_LS(self,c):
        with tf.GradientTape() as t1:
            l = self.loss_fn_LS(c)
        return l,t1.gradient(l,self.u_model.trainable_weights)
    
    
    @tf.function
    def loss_fn_Adam(self,):
        
        x = self.gen_points()
        with tf.GradientTape() as t1:
            t1.watch(x)
            ub = self.u_model(x)
            u = tf.einsum("xi,i->x",ub,self.c)
            
        du = t1.gradient(u,x)
        f = self.rhs(x)
        
        integrand = du**2/2+f*u
                
        return tf.reduce_mean(integrand)
        
    
    @tf.function
    def loss_grads_Adam(self,):
        with tf.GradientTape() as t1:
            l = self.loss_fn_Adam()
        return l,t1.gradient(l,self.u_model.trainable_weights+[self.c])
    
    @tf.function
    def one_step(self,):
        if self.LS:
            c = self.LS_system()
            l,g = self.loss_grads_LS(c)
            self.optimizer.apply_gradients(zip(g,self.u_model.trainable_weights))
        else:
            l,g = self.loss_grads_Adam()
            self.optimizer.apply_gradients(zip(g,self.u_model.trainable_weights+[self.c]))
        return l
    
    def train(self,iterations):
        self.i=0
        while self.i<iterations:
            l = self.one_step()
            self.losslist+= [l]
            self.i+=1
            h1 = self.h1_error()
            print("Epoch:",self.i,"Loss:",float(l),". Err:",float(h1))
            self.h1list +=[float(h1)]


iterations = 5000

neurons =50

npts=25

itslist=[i for i in range(iterations)]


LS_list = [True]

T_list = [TrainingObjects(50,npts,LS=LS) for LS in LS_list]



for T in T_list:
    T.train(iterations)


plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
for T in T_list:
    plt.plot(itslist[::10],T.losslist[::10], label="LS:"+str(T.LS), linewidth=2)
plt.xscale("log")

plt.title("Loss evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Loss", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
for T in T_list:
    plt.plot(itslist[::10],T.h1list[::10], label="LS:"+str(T.LS), linewidth=2)
plt.xscale("log")
plt.yscale("log")

plt.title("Error evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Error", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()


xplot = tf.constant([i/500 for i in range(501)])



plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
for T in T_list:
    if T.LS:
        c=T.LS_system()
    else:
        c=T.c
    ub = T.u_model(xplot)
    u = tf.einsum("xi,i->x",ub,c)
    plt.plot(xplot,u,label="LS:"+str(T.LS))
ue = T_list[0].u_exact(xplot)
plt.plot(xplot,ue,"k--",label="Exact")
plt.legend()
plt.show()
