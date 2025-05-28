# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



class bc_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(bc_layer,self).__init__()
        
    def call(self,inputs):
        
        ##Takes inputs of x, output of other layer
        x,final_layer = inputs
        
        #Define the simple cutoff
        cut = x*(1-x)
        
        #Multiply together. Note, the output is [N,1] shape, 
        #so to make the arithmetic simpler, I "collapse" into [N]
        output = tf.einsum("ij,ij->i",cut,final_layer)
        
        return output



def build_model(neurons,activation):
    
    ##Define input shape
    xvals = tf.keras.layers.Input(shape=(1,), name="x_input")
    
    
    ##A single hiddel layer
    l1 = tf.keras.layers.Dense(neurons, activation=activation)(xvals)
    l2 = tf.keras.layers.Dense(neurons,activation=activation)(l1)

    ##Output
    pre_out = tf.keras.layers.Dense(1)(l2)
    out = bc_layer()([xvals,pre_out])
    
    #Create the model and show information
    model = tf.keras.Model(inputs=xvals, outputs=out)
    model.summary()
    return model

class TrainingObjects:
    def __init__(self, npts, neurons, activation="tanh"):
        
        #Create model
        self.u_model = build_model(neurons,activation)
        self.v_model = build_model(neurons,activation)
        
        #Define optimiser
        self.optimizer_u = tf.keras.optimizers.Adam(learning_rate=10**-3)
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate=10**-2)
        
        #Various lists for saving information
        self.losslist = []
        self.errlist = []
        self.itslist = []
        
        #Quantities for quadrature rule - number of points, element size,
        #integration "mesh"
        self.npts = npts
        self.h=1/npts
        self.x_mesh = tf.constant([i/npts for i in range(npts)]) 
        
        ##We will estimate the error during training. We use a fixed mesh
        self.x_error = tf.constant([(i+0.5)/200 for i in range(200)])
        
        
    ##Define the sampling rule for points - stratified MC
    
    def quad_points(self,):
        uniform = tf.random.uniform([self.npts],maxval=self.h)
        return self.x_mesh+uniform
    
    
    ##Define the right-hand side for the problem 
    def rhs(self,x):
        return -9*np.pi**2*tf.math.sin(3*np.pi*x)
    
    ##The exact solution for comparison
    
    def u_exact(self,x):
        return tf.math.sin(3*np.pi*x)
    
    ##Define the error - we estimate the H^1_0 norm. 
    @tf.function
    def h1_error(self):
        with tf.GradientTape() as t1:
            t1.watch(self.x_error)
            u_error = self.u_model(self.x_error)-self.u_exact(self.x_error)
        du_error = t1.gradient(u_error,self.x_error)
        return tf.reduce_mean(du_error**2)**0.5
    
    ##Define the loss function
    @tf.function
    def loss_fn(self):
        ##Take a sample
        x = self.quad_points()
        
        ##Evaluate function and derivatives
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            u = self.u_model(x)
            v = self.v_model(x)
        du = t1.gradient(u,x)
        dv = t1.gradient(v,x)
        del t1
        
        ##Combine into the loss
        loss = tf.reduce_mean(du*dv+self.rhs(x)*v) -tf.reduce_mean(dv**2)/2
        
        return loss
        
        
    ## Define the function that returns the loss and gradients when evaluated
    ## on the training set. 
    @tf.function
    def loss_grads_u(self,):
        with tf.GradientTape() as t1:
            l = self.loss_fn()
        return l,t1.gradient(l,self.u_model.trainable_weights)
    
    @tf.function
    def loss_grads_v(self,):
        with tf.GradientTape() as t1:
            #Note, we want to maximise over v, optimisers by default 
            #minimise, so we take -loss instead
            l = -self.loss_fn()
        return l,t1.gradient(l,self.v_model.trainable_weights)
    
    
    ##Take the gradient, apply it via the optimiser to update weights and 
    ## return the loss
    @tf.function
    def update_u(self,):
        
        ##Obtain loss and gradient
        l,g = self.loss_grads_u()
        
        ##Apply gradient
        self.optimizer_u.apply_gradients(zip(g,self.u_model.trainable_weights))
        return l
    
    def update_v(self,):
        
        ##Obtain loss and gradient
        l,g = self.loss_grads_v()
        
        ##Apply gradient
        self.optimizer_v.apply_gradients(zip(g,self.v_model.trainable_weights))
        return l
    
    ##Now define the training loop
    def train(self,pre_iterations,iterations,v_steps):
        self.i=0
        
        ##Initial training, only on v
        while self.i < pre_iterations:
            self.i+=1
            l = -self.update_v()
            self.losslist+=[float(l)]
            self.errlist+=[float(self.h1_error())]
            self.itslist+=[self.i]
            
            
        while self.i<iterations+pre_iterations:
            
            self.i+=1
            for j in range(v_steps):
                l = -self.update_v()
            
            ##This evaluates the loss, updates model via gradients. 
            l = self.update_u()
            
            ##Record the training and validation loss
            self.losslist+= [float(l)]
            error = self.h1_error()
            self.errlist += [float(error)]
            self.itslist+=[self.i]
            
            
            ##Print information 
            print("Epoch:",self.i,"Loss:",float(l),"Error:",float(error))



pre_iterations=1000

#Somewhat stable choice.
# iterations = 1000
# v_step = 10

##Unstable choice
iterations =5000
v_step=1

neurons =25




history_list = []


size_list = []

T = TrainingObjects(250,neurons)
T.train(pre_iterations,iterations,v_step)




###Show the evolution of loss
plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
plt.plot(T.itslist[::100],T.losslist[::100], label="Training", linewidth=2)  # Set line width for better visibility
plt.xscale("log")


plt.title("Loss evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Loss", fontsize=12)  # Label for y-axis
plt.legend( loc="best", fontsize=10)  # Clearer legend
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

###Show the evolution of error
plt.figure(figsize=(4, 3),dpi=200)  # Set figure size
plt.plot(T.itslist[::100],T.errlist[::100], label="H^1_0 error", linewidth=2)  # Set line width for better visibility
plt.xscale("log")
plt.yscale("log")

plt.title("Error evolution")
plt.xlabel("Iterations", fontsize=12)  # Label for x-axis
plt.ylabel("Error", fontsize=12)  # Label for y-axis
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

xplot = tf.constant([i/100 for i in range(101)])


ue =T.u_exact(xplot)

u_final = T.u_model(xplot)

plt.figure(figsize=(4, 3),dpi=200) 
plt.plot(xplot, ue, label="Exact", linewidth=2, color="#2ca02c")
plt.plot(xplot, u_final, label="Final", linewidth=2, linestyle="--", color="#1f77b4")

plt.xlabel("x", fontsize=13)
plt.ylabel("u(x)", fontsize=13)

plt.tick_params(labelsize=11)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 3),dpi=200) 
plt.plot(xplot, ue-u_final, label="Error", linewidth=2, color="#2ca02c")

plt.xlabel("x", fontsize=13)
plt.ylabel("error", fontsize=13)

plt.tick_params(labelsize=11)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.show()


plt.figure(figsize=(4, 3),dpi=200) 
plt.scatter(T.losslist,T.errlist)
plt.xscale("log")
plt.yscale('log')
plt.xlabel("Loss")
plt.ylabel("H^1 error")

plt.tight_layout()
plt.show()


