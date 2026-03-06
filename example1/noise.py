#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 00:09:59 2025

@author: liaochunyang
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

def left_odd_ext(xleft,x):
    Lext_x = 2*xleft - x
    return Lext_x[::-1]
    
def right_odd_ext(xright,x):
    Rext_x = 2*xright - x
    return  Rext_x[::-1]

def extend_sol_matrix(t, sol_matrix, ext_amount, n_initial_cond):
    tleft = t[0]
    Lext_t = (2*tleft - t[1:ext_amount])[::-1]
    tright = t[-1]
    Rext_t = (2*tright - t[-ext_amount:-1])[::-1]
    xleft = sol_matrix[0,0]
    xright = sol_matrix[0,-1]
    Lxsimm = sol_matrix[0,1:ext_amount]
    Rxsimm = sol_matrix[0,-ext_amount:-1]
    Lext_sol = left_odd_ext(xleft,Lxsimm)
    Rext_sol = right_odd_ext(xright,Rxsimm)
    
    left_ext = Lext_sol
    right_ext = Rext_sol
    for i in range(1,n_initial_cond):
        xleft = sol_matrix[i,0]
        xright = sol_matrix[i,-1]
        Lxsimm = sol_matrix[i,1:ext_amount]
        Rxsimm = sol_matrix[i,-ext_amount:-1]
        
        Lext_sol = left_odd_ext(xleft,Lxsimm)
        left_ext = np.vstack((left_ext,Lext_sol))
        
        Rext_sol = right_odd_ext(xright,Rxsimm)
        right_ext = np.vstack((right_ext,Rext_sol))
    extended_t = np.hstack((Lext_t,t, Rext_t))
    extended_approx_solut = np.hstack((left_ext,sol_matrix,right_ext))
    return extended_t, extended_approx_solut

def add_noise(X, t, deltat, noise_level):
    """
    Add noise to approximated solution, and then smooth noisy data with splines.
    """
    n_initial_cond = X.shape[0]
    # compute trajectories range as (max(traj)-min(traj)) then compute mea across all trajectories
    #This is the mean solution range
    maxmin=[]
    for i in range(n_initial_cond):
        maxmin.append(np.max(X[i,:])-np.min(X[i,:]))
    mean_range = np.mean(maxmin)
    
    # add noise and extend the boundary
    X_noise = X + np.random.normal(0.0, noise_level, (n_initial_cond,len(t))) * mean_range
    ext_amount = 2
    extended_T, extended_X_noise = extend_sol_matrix(t, X, ext_amount, n_initial_cond)
    
    #smooth noisy data with splines to generate good difference quotients
    X_spline = np.zeros((n_initial_cond,len(extended_T)))
    for i in range(n_initial_cond):
        y = extended_X_noise[i,:]
        spl = UnivariateSpline(extended_T, y, k = 4) 
        spl.set_smoothing_factor(0.5) #increase or decrease smoothness of spline (1: 1e06 2:0.5)
        X_spline[i,:] = spl(extended_T)
   
    #compute new difference quotients from smoothed data
    n_ext_times = len(extended_T)
    DQ_noise = np.zeros((n_initial_cond,n_ext_times))
    for i in range(n_initial_cond):
        for k in range(n_ext_times-1):
            DQ_noise[i,k] = (X_spline[i,k+1] - X_spline[i,k])/deltat #difference quotients
        DQ_noise[i,n_ext_times-1] = X_spline[i, n_ext_times-1] * np.cos(X_spline[i, n_ext_times-1]) 
        
    return X_noise, DQ_noise