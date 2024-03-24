#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code implements a simple parametric Physics Informed Neural Network (PINN) to solve the damped harmonic oscillator problem.
A damped harmonic oscillator is described by the second-order ordinary differential equation (ODE):

    u_tt + 2 * ξ * u_t + u = 0

where:
  - u_tt is the second-order derivative of u with respect to t (acceleration)
  - u_t is the first-order derivative of u with respect to t (velocity)
  - u is the displacement (position) 
  - ξ (or xi) is the damping coefficient, and t is time.

This equation describes the motion of a damped harmonic oscillator,
where the acceleration u_tt is balanced by the damping force -2 * xi * u_t
and the restoring force -u, resulting in a net force of 0.
PINN is a deep learning-based approach that combines physics-based constraints (ODE) with data-driven learning 
to approximate the solution to the differential equation.

@author: mabedi
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

#%% Utilities

def build_model(n_neurons, activation='tanh'):
    # Define the PINN model architecture with two input layers
    initializer = tf.random_normal_initializer(0., 0.02, seed=0)

    # First input layer for time points (t_points)
    t_points = Input(shape=(1,), name='x_input')
    l1a = Dense(n_neurons, activation=activation, name='l1a', kernel_initializer=initializer, use_bias=True)(t_points)
    l2a = Dense(n_neurons*2, activation=activation, name='l2a', kernel_initializer=initializer, use_bias=True)(l1a)
    l3a = Dense(n_neurons*2, activation=activation, name='l3a', kernel_initializer=initializer, use_bias=True)(l2a)

    # Second input layer for ξ the damping coefficient  (xi)
    Xi = Input(shape=(1,), name='xi_input')
    l1b = Dense(n_neurons, activation=activation, name='l1b', kernel_initializer=initializer, use_bias=True)(Xi)
    l2b = Dense(n_neurons*2, activation=activation, name='l2b', kernel_initializer=initializer, use_bias=True)(l1b)
    l3b = Dense(n_neurons*2, activation=activation, name='l3b', kernel_initializer=initializer, use_bias=True)(l2b)
    
    # Combine the outputs from the two branches
    l7 = tf.add(l3a, l3b)
    
    # Additional layers
    l8 = Dense(n_neurons*2, activation=activation, name='l8', dtype=tf.float32, kernel_initializer=initializer, use_bias=True)(l7)
    output = Dense(1, activation=None, name='lout', dtype=tf.float32, kernel_initializer=initializer, use_bias=True)(l8)

    # Define the model with multiple inputs and one output
    u_model = Model(inputs=[t_points, Xi], outputs=output, name='u_model')
    u_model.summary()

    return u_model

@tf.function
def compute_gradients(model, t_points, xi):
    # Compute gradients for the PINN loss
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_points)
        u = model([t_points, xi])  # Pass both inputs to the model
        u_t = tape.gradient(u, t_points) # 1st order
    u_tt = tape.gradient(u_t, t_points) # 2nd order
    return u, u_t, u_tt

@tf.function
def loss_pinn(model, t0, x0, v0, t_points, xi):
    u, u_t, u_tt = compute_gradients(model, t_points, xi)
    u0, u_t0, _ = compute_gradients(model, t0, xi)  # Pass t0 as an input
    
    # Define the PINN loss function
    ode_loss = u_tt + 2. * xi * u_t + u 
    IC_loss_1 = u0 - x0 # initial condition
    IC_loss_2 = u_t0 - v0 # initial condition
    
    ODE_term = tf.reduce_mean(tf.math.pow(ode_loss, 2)) 
    IC1_term = tf.reduce_mean(tf.math.pow(IC_loss_1, 2)) 
    IC2_term = tf.reduce_mean(tf.math.pow(IC_loss_2, 2))
    
    total_loss = ODE_term + (IC1_term + IC2_term)
    metrics = ODE_term, IC1_term, IC2_term
    
    return total_loss, metrics

@tf.function
def train_step(model, optimizer, t0, x0, v0, t_points, xi):
    # Define a training step for the PINN model
    with tf.GradientTape() as tape:
        loss, metrics = loss_pinn(model, t0, x0, v0, t_points, xi)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, metrics

#%% Training

# Define the PINN model
model = build_model(n_neurons=32)

num_epochs = 100000 # Number of epochs

# Define initial conditions
x0 = tf.constant([[2.0]])  # Initial position
t0 = tf.constant([[0.0]])  # Initial time
v0 = tf.constant([[1.0]])  # Initial velocity

npt = 100 #Number of time points used for calculation of time steps 
min_time = 0.
max_time = 25.

# Define optimizer with decreasing learning rate
initial_learning_rate = 0.002
final_learning_rate = 0.0001
def lr_schedule(epoch):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer,loss=loss_pinn)

# Training loop initialization
start_time0 = time.time()
start_time = start_time0
loss_total = []
metrics_total_ODE = []
metrics_total_IC1 = []
metrics_total_IC2 = []
loss0 = float('inf')

# Training loop
for epoch in range(num_epochs):
    #Changing the learning rate
    new_learning_rate = lr_schedule(epoch)
    optimizer.learning_rate.assign(new_learning_rate)
    
    # Generate random time points
    t_points = tf.random.uniform([npt, 1], minval=min_time, maxval=max_time, dtype=tf.float32, seed=epoch)
    
    # Generate a random damping coefficient (xi)
    xi = tf.random.uniform([1, 1], minval=0.01, maxval=0.5, dtype=tf.float32, seed=epoch)
    
    # Perform one training step
    loss, metrics = train_step(model, optimizer, t0, x0, v0, t_points, xi)
    
    # Record loss and metrics
    loss_total.append(loss)
    metrics_total_ODE.append(metrics[0])
    metrics_total_IC1.append(metrics[1])
    metrics_total_IC2.append(metrics[2])
    
    # Print progress
    if epoch % 100 == 0:
        print("Epoch %d, Loss = %.6f, Time taken: %.2fs, xi = %.3f" % (epoch, float(loss), time.time() - start_time, float(tf.reduce_min(xi))))
        start_time = time.time()
        
        # Save the model if the loss decreases
        if loss < loss0:
            model_saved = model
            loss0 = loss

print("Total training time: %.3fs" % (time.time() - start_time0))

# Saving the trained model
# model_saved.save('model_saved')

# Plotting the evolution of Loss and each term of the loss as metrics:
plt.figure()
plt.plot(tf.squeeze(loss_total), 'k.')
plt.title('Total loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.yscale('log')

plt.figure()
plt.plot(tf.squeeze(metrics_total_ODE), 'g.')
plt.plot(tf.squeeze(metrics_total_IC1), 'r.')
plt.plot(tf.squeeze(metrics_total_IC2), 'b.')
plt.legend(['PINN','1st initial condition','2nd initial condition'])
plt.title('Each term of loss')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.yscale('log')

#%% Example of usage

#Parameters that can be changed after training:
npt = 250 # Number of test time points
xi=0.2   # ξ in inference stage

test_t_points = tf.expand_dims(np.linspace(min_time, max_time, npt), axis=1)  

# The exact solution on test time points
wd = np.sqrt(1 - xi**2)
C = np.sqrt(x0**2 + ((v0 + xi*x0/wd)**2))
phi = np.arctan(x0**wd / (v0 + xi*x0))
true_u = C * np.exp(-xi * test_t_points) * np.sin(wd * test_t_points + phi)  

# Loading and applying the trained model
# model_saved=tf.keras.models.load_model('model_saved', custom_objects={'loss_pinn': loss_pinn})
pred_u = model_saved([test_t_points, tf.constant([[xi]])])

# Plotting results
font = {'family' : 'Times',
        'size'   : 17}
plt.rc('font', **font)
plt.figure()
plt.plot(test_t_points, true_u, 'k')
plt.plot(test_t_points, pred_u, '--r' )
plt.legend(['Explicit solution','PINN Prediction'])
plt.xlabel('t')
plt.ylabel('u')
plt.show()   
    