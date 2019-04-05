#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:24:41 2019

@author: graceswenson
"""

import numpy as np
from numpy.linalg import norm , eigh
import matplotlib.pyplot as plt



#PART 1 : FUNCTION RETURNING FREQUENCIES AND EIGENVECTORS

def calc_frequencies_and_modes(matrix , k_over_m):
    M = (k_over_m) * matrix
    
    eigval , eigvec = eigh(M)
    
    frequencies = np.sqrt(eigval)
    
    normvec = eigvec
    
    return frequencies , normvec

#PART 2: CALCULATING COEFFICIENTS FOR INITIAL CONDITIONS

def calc_components_from_initial_conditions(x_init , modes):
    
    ve1 = modes[:,0]
    
    ve2 = modes[:,1]
    
    a = x_init @ ve1
    
    b = x_init @ ve2
    
    return a , b

#PART 3:
    
def time_depend(a , b , t , frequencies , normvec):
    
    x_t = a * np.cos(frequencies[0] * t) * normvec[:,0] + b * np.cos(frequencies[1] * t) * normvec[:,1]
    
    return x_t

def plot_motion_of_masses(x, time, title='Motion of Masses'):
    """
    Function to make a plot of motion of masses as a function of time. The time
    should be on the vertical axis and the position on the horizontal axis.
    Parameters
    ----------
    x : array of position, N_times by 2 elements
        The array of positions, set up so that x[:, 0] is the position of mass
        1 relative to equilibrium and x[:, 1] is the position of mass 2.
    time : array of times
        Times at which the positions have been calculated.
    title : str
        A descriptive title for the plot to make grading easier.
    """
    x1_equilibrium_pos = 3
    x2_equilibrium_pos = 6

    x1 = x[:, 0] + x1_equilibrium_pos
    x2 = x[:, 1] + x2_equilibrium_pos

    plt.plot(x1, time, label='Mass 1')
    plt.plot(x2, time, label='Mass 2')
    plt.xlim(0, 9)
    plt.legend()


matrix1 = np.array([[2 , -1],
                   [-1 , 2]])
    
frequencies1 , normvec1 = calc_frequencies_and_modes(matrix1 , 1)

#checked for each of the two eigen vectors, then did Part 4: with two other vectors and one of our choosing

#x_init = np.array([1,1])
#x_init = np.array([1 , -1])
#x_init = np.array([1,0])
#x_init = np.array([0,1])
x_init = np.array([0 , -1])


a1 , b1 = calc_components_from_initial_conditions(x_init , normvec1)

t_init1 = 0
t_end1 = 10
N_times1 = 1000

time1 = np.linspace(t_init1 , t_end1 , num = N_times1)

time1 = time1.reshape(N_times1 , 1)


#print(time_depend(a1 , b1 , time1 , frequencies1 , normvec1))

    
plot_motion_of_masses(time_depend(a1,b1,time1,frequencies1,normvec1) , time1)
plt.show()

#PART 5: ANOTHER MASS-SPRING SYSTEM

matrix2 = np.array([[2 , -1],
                    [-1/10 , 1/5]])
    
frequencies2 , normvec2 = calc_frequencies_and_modes(matrix2 , 1)

#x_init2 = np.array([1,1])
#x_init2 = np.array([1 , -1])
#x_init2 = np.array([1 , 0])
x_init2 = np.array([0 , 1])


a2 , b2 = calc_components_from_initial_conditions(x_init2 , normvec2)

t_init2 = 0
t_end2 = 10
N_times2 = 1000

time2 = np.linspace(t_init2 , t_end2 , num = N_times2)

time2 = time2.reshape(N_times2 , 1)

#print("new mass-spring system = " , time_depend(a2 , b2 , time2 , frequencies2 , normvec2))

plot_motion_of_masses(time_depend(a2 , b2 , time2 , frequencies2 , normvec2) , time2)

#MASSES MOVE DIFFERENTLY AT DIFFERENT MASSES. THE HEAVIER ONE MOVES SLOWER, AND
#   THUS WOULD HAVE TO BE PUSHED/PULLED WITH A LARGER FORCE TO GET IT TO MOVE 
#   AROUND THE SAME VELOCITY AS THE LIGHTER ONE
