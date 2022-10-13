#!/usr/bin/python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import mujoco as mj

def lqr(model, data, qpos0, ctrl0):    
    nu = model.nu  # Alias for the number of actuators.
    R = np.eye(nu)

    nv = model.nv  # Shortcut for the number of DoFs.

    # Get the Jacobian for the root body (torso) CoM.
    mj.mj_resetData(model, data)
    data.qpos = qpos0
    mj.mj_forward(model, data)
    jac_com = np.zeros((3, nv))
    torso_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'torso')
    mj.mj_jacSubtreeCom(model, data, jac_com, torso_body_id)

    # Get the Jacobian for the left foot.
    jac_foot = np.zeros((3, nv))
    foot_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'foot_left')
    mj.mj_jacBodyCom(model, data, jac_foot, None, foot_body_id)

    jac_diff = jac_com - jac_foot
    Qbalance = jac_diff.T @ jac_diff

    # Get all joint names.
    joint_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # Get indices into relevant sets of joints.
    root_dofs = range(6)
    body_dofs = range(6, nv)
    abdomen_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'abdomen' in name
        and not 'z' in name
    ]
    left_leg_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'left' in name
        and ('hip' in name or 'knee' in name or 'ankle' in name)
        and not 'z' in name
    ]
    balance_dofs = abdomen_dofs + left_leg_dofs
    other_dofs = np.setdiff1d(body_dofs, balance_dofs)

    # Cost coefficients.
    BALANCE_COST        = 1000  # Balancing.
    BALANCE_JOINT_COST  = 3     # Joints required for balancing.
    OTHER_JOINT_COST    = .3    # Other joints.

    # Construct the Qjoint matrix.
    Qjoint = np.eye(nv)
    Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
    Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
    Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

    # Construct the Q matrix for position DoFs.
    Qpos = BALANCE_COST * Qbalance + Qjoint

    # Qpos = np.eye(nv)

    # No explicit penalty for velocities.
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                [np.zeros((nv, 2*nv))]])


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

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    data.qpos = qpos0

    # Allocate position difference dq.
    dq = np.zeros(model.nv)

    return K, dq