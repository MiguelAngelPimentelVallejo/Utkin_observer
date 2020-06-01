#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""  This script is to simulate a example 6.2.2 in book 
Sliding Mode Cotrol Theory and Applications by Christopher Edwars
where explained a Utikin Observer """

__author__ = '{Miguel Angel Pimentel Vallejo}'
__email__ = '{miguel.pimentel@umich.mx}'
__date__= '{14/may/2020}'

#Import the modules needed to run the script.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

#Fuction tha contains the script model
def obs_model(t,x):
    
    #Linear model parameters
    A = np.matrix([[0,1],[-2,0]])
    B = np.matrix([[0],[1]])
    C = np.matrix([[1,1]])

    #Observer paramters
    Tc = np.matrix([[-1,1],[1,1]])
    M = 1
    L = -1

    #Matrix operation to obtain the other parameters of the observer
    TATI = Tc*A*np.linalg.inv(Tc)
    TB = Tc*B
    
    A_til_11 = TATI[0,0] + L*TATI[1,0]
    A_til_12 = TATI[0,1] + L*TATI[1,1] - A_til_11*L
    A_til_22 = TATI[1,1] - TATI[1,0]*L

    #Declare the list to contain the derivative result
    # xdot position -> derivative meaning
    #[0] -> \.x_1, 
    #[1] -> \.y, 
    #[2] -> \.\^x_1, 
    #[3] -> \.\^y,
    #[4] -> \.\~e_1,
    #[5] -> \.e_y.
    xdot = [0,0,0,0,0,0]

    #Input
    u = 0

    #Errors
    e_1 = x[2] - x[0]
    e_y = x[3] - x[1]

    #Error with respect new coordinates
    e_til_1 = e_1 + L*x[1]

    #Discontinouos vector
    v = M*np.sign(e_y)

    #Calculus of the dynamics of the system
    xdot[0] = TATI[0,0]*x[0] + TATI[0,1]*x[1] + TB[0,0]*u 
    xdot[1] = TATI[1,0]*x[0] + TATI[1,1]*x[1] + TB[1,0]*u
    xdot[2] = TATI[0,0]*x[2] + TATI[0,1]*x[3] + TB[0,0]*u + L*v
    xdot[3] = TATI[1,0]*x[2] + TATI[1,1]*x[3] + TB[1,0]*u - v
    xdot[4] = A_til_11*e_til_1 + A_til_12*e_y
    xdot[5] = TATI[1,0]*e_til_1 + A_til_22*e_y - v

    return xdot

#Fuction tha contains the script model with uncertainty
def obs_model_uncer(t,x):
    
    #Linear model parameters
    A = np.matrix([[0,1],[-2,0]])
    B = np.matrix([[0],[1]])
    C = np.matrix([[1,1]])

    #Uncertainty matrix
    A_uncer = np.matrix([[0.2,0.8],[-2.1,0.25]])

    #Observer paramters
    Tc = np.matrix([[-1,1],[1,1]])
    M = 1
    L = -1

    #Matrix operation to obtain the other parameters of the observer
    TATI = Tc*A*np.linalg.inv(Tc)
    TB = Tc*B
    
    A_til_11 = TATI[0,0] + L*TATI[1,0]
    A_til_12 = TATI[0,1] + L*TATI[1,1] - A_til_11*L
    A_til_22 = TATI[1,1] - TATI[1,0]*L

    #Matrix operations with uncertainy
    TATI_uncer = Tc*A_uncer*np.linalg.inv(Tc)

    A_til_11_uncer = TATI_uncer[0,0] + L*TATI_uncer[1,0]
    A_til_12_uncer = TATI_uncer[0,1] + L*TATI_uncer[1,1] - A_til_11*L
    A_til_22_uncer = TATI_uncer[1,1] - TATI_uncer[1,0]*L

    #Declare the list to contain the derivative result
    # xdot position -> derivative meaning
    #[0] -> \.x_1, 
    #[1] -> \.y, 
    #[2] -> \.\^x_1, 
    #[3] -> \.\^y,
    #[4] -> \.\~e_1,
    #[5] -> \.e_y.
    xdot = [0,0,0,0,0,0]

    #Input
    u = 0

    #Errors
    e_1 = x[2] - x[0]
    e_y = x[3] - x[1]

    #Error with respect new coordinates
    e_til_1 = e_1 + L*x[1]

    #Discontinouos vector
    v = M*np.sign(e_y)

    #Calculus of the dynamics of the system
    xdot[0] = TATI[0,0]*x[0] + TATI[0,1]*x[1] + TB[0,0]*u 
    xdot[1] = TATI[1,0]*x[0] + TATI[1,1]*x[1] + TB[1,0]*u
    xdot[2] = TATI_uncer[0,0]*x[2] + TATI_uncer[0,1]*x[3] + TB[0,0]*u + L*v
    xdot[3] = TATI_uncer[1,0]*x[2] + TATI_uncer[1,1]*x[3] + TB[1,0]*u - v
    xdot[4] = A_til_11_uncer*e_til_1 + A_til_12_uncer*e_y
    xdot[5] = TATI_uncer[1,0]*e_til_1 + A_til_22_uncer*e_y - v

    return xdot

#Initials condictions for the system
x0 = [1,-1]

#Initials condictions for the observer
xo0 = [0,0]

#Intial error
e0 = xo0[0]-x0[0]
e1 = xo0[1]-x0[1]

#Initial condictions vector
x0 = [x0[0],x0[1],xo0[0],xo0[1],e0,e1]

#Initial time
t0 = 0

#ODE with Runge Kutta
r = ode(obs_model).set_integrator('dopri5',atol=1.e-3,rtol=1.e-3)
r.set_initial_value(x0, t0)

#Final time
tf = 15

#Step size
dt = 0.006

#Create list to save the result of the solver
x1 = [x0[0]]
x2 = [x0[1]]
x3 = [x0[2]]
x4 = [x0[3]]
x5 = [x0[4]]
x6 = [x0[5]]
t = [t0]
M = 1
v = [M*np.sign(x2[0] - x4[0])]

#Loop to solve the ODE
while r.successful() and r.t < tf:
    r.t+dt
    r.integrate(r.t+dt)
    x1.append(r.y[0])
    x2.append(r.y[1])
    x3.append(r.y[2])
    x4.append(r.y[3])
    x5.append(r.y[4])
    x6.append(r.y[5])
    t.append(r.t)
    v.append(M*np.sign(r.y[1] - r.y[3]))

        
#plot results

#Figure 1 plot the system vs observer
labels = ['$x_1$','$y$','$\^x_1$','$\^y$']
plt.figure()
plt.title('System and observer')
est1 = plt.plot(t,x1)
est2 = plt.plot(t,x2)
ob1 = plt.plot(t,x3,'--')
ob2 = plt.plot(t,x4,'--')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(est1 + est2 + ob1 + ob2,labels)

#Figure 2 plot the error with respect new coordinates
labels = ['$ \~e_1 $','$e_y$']
plt.figure()
plt.title('Error system with respect new coordinates')
est5 = plt.plot(t,x5)
est6 = plt.plot(t,x6)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(est5 + est6 ,labels)

#Figure 3 plot the discontinuous vector
plt.figure()
labels = ['$v$']
plt.title('Discontinuous vector')
v_l = plt.plot(t,v)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(v_l ,labels)

#Figure 4 plot the error bewteen the observer and system
plt.figure()
labels = ['$e_1$', '$e_y$']
plt.title('Error system')
e_1_l = plt.plot(t, (np.matrix(x3)-np.matrix(x1)).tolist()[0])
e_y_l = plt.plot(t, (np.matrix(x4)-np.matrix(x2)).tolist()[0])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(e_1_l + e_y_l ,labels)

#ODE with Runge Kutta
r = ode(obs_model_uncer).set_integrator('dopri5',atol=1.e-3,rtol=1.e-3)
r.set_initial_value(x0, t0)

#Final time
tf = 30

#Step size
dt = 0.006

#Create list to save the result of the solver
x1 = [x0[0]]
x2 = [x0[1]]
x3 = [x0[2]]
x4 = [x0[3]]
x5 = [x0[4]]
x6 = [x0[5]]
t = [t0]
M = 1
v = [M*np.sign(x2[0] - x4[0])]

#Loop to solve the ODE
while r.successful() and r.t < tf:
    r.t+dt
    r.integrate(r.t+dt)
    x1.append(r.y[0])
    x2.append(r.y[1])
    x3.append(r.y[2])
    x4.append(r.y[3])
    x5.append(r.y[4])
    x6.append(r.y[5])
    t.append(r.t)
    v.append(M*np.sign(r.y[1] - r.y[3]))

        
#plot results

#Figure 1 plot the system vs observer
labels = ['$x_1$','$y$','$\^x_1$','$\^y$']
plt.figure()
plt.title('System and observer')
est1 = plt.plot(t,x1)
est2 = plt.plot(t,x2)
ob1 = plt.plot(t,x3,'--')
ob2 = plt.plot(t,x4,'--')
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(est1 + est2 + ob1 + ob2,labels)

#Figure 2 plot the error with respect new coordinates
labels = ['$ \~e_1 $','$e_y$']
plt.figure()
plt.title('Error system with respect new coordinates')
est5 = plt.plot(t,x5)
est6 = plt.plot(t,x6)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(est5 + est6 ,labels)

#Figure 3 plot the discontinuous vector
plt.figure()
labels = ['$v$']
plt.title('Discontinuous vector')
v_l = plt.plot(t,v)
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(v_l ,labels)

#Figure 4 plot the error bewteen the observer and system
plt.figure()
labels = ['$e_1$', '$e_y$']
plt.title('Error system')
e_1_l = plt.plot(t, (np.matrix(x3)-np.matrix(x1)).tolist()[0])
e_y_l = plt.plot(t, (np.matrix(x4)-np.matrix(x2)).tolist()[0])
plt.xlabel('time')
plt.ylabel('magnitude')
plt.grid(True)
plt.legend(e_1_l + e_y_l ,labels)

plt.show()