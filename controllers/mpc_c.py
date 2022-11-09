#!/usr/bin/python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import mujoco as mj

from controllers.mpc import LinearMPC

def mpc(model, data, qpos0, ctrl0):    
    nu = model.nu  # Alias for the number of actuators.
    R = np.eye(nu)

    nv = model.nv  # Shortcut for the number of DoFs.
    Qpos = 300* np.eye(nv)

    # No explicit penalty for velocities.
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                [np.zeros((nv, nv)), Qpos]])

    Q = 300* np.eye(2*nv)


    # Set the initial state and control.
    mj.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    # Allocate the A and B matrices, compute them.
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    centered = True
    mj.mjd_transitionFD(model, data, epsilon, centered, A, B, None, None)

    return A,B,Q,R, nv