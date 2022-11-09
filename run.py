import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import scipy.linalg
from mujoco_base import MuJoCoBase
from controllers.lqr import lqr
from controllers.mpc_c import mpc
from controllers.mpc import LinearMPC


class Biped(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.sim_end = 110.0  #End time of simulation

        self.arms = None#"fixed"

    def reset(self, model, data):
        # Set camera configuration
        self.cam.azimuth = 120.89  # 89.608063
        self.cam.elevation = -15.81  # -11.588379
        self.cam.distance = 8.0  # 5.0
        self.cam.lookat = np.array([0.0, 0.0, 2.0])

        # # Stand on 1 Leg
        # qpos0, ctrl0 = self.stand_one_leg(model, data)

        # Stand on 2 legs
        qpos0, ctrl0 = self.stand_two_legs(model, data)

        # Get the Controller Value K
        K, dq = lqr(model, data, qpos0, ctrl0)
        A,B,Q,R, nv = mpc(model, data, qpos0, ctrl0)
        LMPC =  LinearMPC(A,B,Q,R)

        # Perturbations 
        CTRL_STD, perturb = self.noise(model)

        return qpos0, ctrl0, K, dq, CTRL_STD, perturb, LMPC, nv

    def controller(self, model, data, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step, LMPC, nv):
        """
        
        """
        # Get state difference dx.
        mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T

        # LQR control law.
        # data.ctrl = ctrl0 - K @ dx
        
        # MPC control law.
        gs = np.zeros((11,2*nv))
        data.ctrl = ctrl0 + LMPC.obtain_sol(dx, gs)
        print(np.linalg.norm(dx))

        if self.arms == "passive":
            data.ctrl[15] = 0
            data.ctrl[16] = 0
            data.ctrl[17] = 0
            data.ctrl[18] = 0
            data.ctrl[19] = 0
            data.ctrl[20] = 0

        elif self.arms == "fixed":
            data.ctrl[15] = ctrl0[15]
            data.ctrl[16] = ctrl0[16]
            data.ctrl[17] = ctrl0[17]
            data.ctrl[18] = ctrl0[18]
            data.ctrl[19] = ctrl0[19]
            data.ctrl[20] = ctrl0[20]


        # # Add perturbation, increment step.
        # data.ctrl += CTRL_STD*perturb[step]

    def simulate(self):
        step = 0
        qpos0, ctrl0, K, dq, CTRL_STD, perturb, LMPC, nv = \
        self.reset(self.model, self.data)
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            # if self.data.time >= 10 and self.data.time <= (10 + 0.2):
            #     self.apply_external_forces([50.0, 0.0, 0.0])
            # else:
            #     self.apply_external_forces([0.0, 0.0, 0.0])

            
            x = 10*int(self.data.time/10)
            y = x + 0.2
            if self.data.time >= x and self.data.time <= y:
                self.apply_external_forces([55.0 + 2*x/10, 0.0, 0.0])
            else:
                self.apply_external_forces([0.0, 0.0, 0.0])
            
            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

                # Apply control
                self.controller(self.model, self.data, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step, LMPC, nv)
                step +=1
                if step >=2390:
                    step = 0

            
            if self.data.time >= self.sim_end:
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 0

            # Enable perturbation force visualisation.
            self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = 1

            # Enable contact force visualisation.
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            # Set the scale of visualized contact forces to 1cm/N.
            self.model.vis.map.force = 0.01

            # Update scene and render
            self.cam.lookat[0] = self.data.qpos[0]
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()

    def quat2euler(self, quat):
        # SciPy defines quaternion as [x, y, z, w]
        # MuJoCo defines quaternion as [w, x, y, z]
        _quat = np.concatenate([quat[1:], quat[:1]])
        r = R.from_quat(_quat)

        # roll-pitch-yaw is the same as rotating w.r.t
        # the x, y, z axis in the world frame
        euler = r.as_euler('xyz', degrees=False)

        return euler

    def stand_one_leg(self, model, data):
        mj.mj_resetDataKeyframe(model, data, 1)
        mj.mj_forward(model, data)
        # print(data.qacc)
        data.qacc = 0  # Assert that there is no the acceleration.
        # print(data.qacc)
        mj.mj_inverse(model, data)
        # print(data.qfrc_inverse)

        height_offsets = np.linspace(-0.001, 0.001, 2001)
        vertical_forces = []
        for offset in height_offsets:
            mj.mj_resetDataKeyframe(model, data, 1)
            mj.mj_forward(model, data)
            data.qacc = 0
            # Offset the height by `offset`.
            data.qpos[2] += offset
            mj.mj_inverse(model, data)
            vertical_forces.append(data.qfrc_inverse[2])

        # Find the height-offset at which the vertical force is smallest.
        idx = np.argmin(np.abs(vertical_forces))
        best_offset = height_offsets[idx]

        mj.mj_resetDataKeyframe(model, data, 1)
        mj.mj_forward(model, data)
        data.qacc = 0
        data.qpos[2] += best_offset
        qpos0 = data.qpos.copy()  # Save the position setpoint.
        mj.mj_inverse(model, data)
        qfrc0 = data.qfrc_inverse.copy()
        print('desired forces:', qfrc0)

        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
        ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
        print('control setpoint:', ctrl0)

        mj.mj_resetData(model, data)
        data.qpos = qpos0
        data.ctrl = ctrl0

        return qpos0, ctrl0

    def stand_two_legs(self, model, data):
        mj.mj_resetDataKeyframe(model, data, 2)
        mj.mj_forward(model, data)
        data.qacc = 0  # Assert that there is no the acceleration.
        mj.mj_inverse(model, data)

        height_offsets = np.linspace(-0.001, 0.001, 2001)
        vertical_forces = []
        for offset in height_offsets:
            mj.mj_resetDataKeyframe(model, data, 2)
            mj.mj_forward(model, data)
            data.qacc = 0
            # Offset the height by `offset`.
            data.qpos[2] += offset
            mj.mj_inverse(model, data)
            vertical_forces.append(data.qfrc_inverse[2])

        # Find the height-offset at which the vertical force is smallest.
        idx = np.argmin(np.abs(vertical_forces))
        best_offset = height_offsets[idx]

        mj.mj_resetDataKeyframe(model, data, 2)
        mj.mj_forward(model, data)
        data.qacc = 0
        data.qpos[2] += best_offset
        qpos0 = data.qpos.copy()  # Save the position setpoint.
        mj.mj_inverse(model, data)
        qfrc0 = data.qfrc_inverse.copy()

        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
        ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.

        mj.mj_resetData(model, data)
        data.qpos = qpos0
        data.ctrl = ctrl0

        return qpos0, ctrl0



    def noise(self, model):
        nu = model.nu  # Alias for the number of actuators.

        DURATION = 12         # seconds
        CTRL_STD = 0.05       # actuator units
        CTRL_RATE = 0.8       # seconds

        # Precompute some noise.
        np.random.seed(1)
        nsteps = int(np.ceil(DURATION/model.opt.timestep))
        perturb = np.random.randn(nsteps, nu)

        # Smooth the noise.
        width = int(nsteps * CTRL_RATE/DURATION)
        kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
        kernel /= np.linalg.norm(kernel)
        for i in range(nu):
            perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')

        return CTRL_STD, perturb

    def apply_external_forces(self, force):

        self.data.qfrc_applied[0] = force[0]
        self.data.qfrc_applied[1] = force[1]
        self.data.qfrc_applied[2] = force[2]

        # Position and size of perturbation-visualisation capsule.
        bpos = self.data.xipos[1]
        ppos = bpos + force
        self.model.geom_size[20][1] = np.linalg.norm(force)/400
        # Make visualisation capsule visible.
        self.model.geom_rgba[20][3] = 1

        # Set position / orientation of perturbation-visualisation capsule.
        quat = np.zeros(4)
        mat = np.zeros(9)
        pertnorm = force / (np.linalg.norm(force)+0.0001)
        mj.mju_quatZ2Vec(quat, pertnorm)
        mj.mju_quat2Mat(mat, quat)
        self.data.geom_xpos[20] = ppos
        self.data.geom_xmat[20] = mat

        if np.linalg.norm(np.array(force)) != 0:
            print("Force is applied:",force[0], self.data.qfrc_applied[0]) 
            
            # Make visualisation capsule visible.
            self.model.geom_rgba[20][3] = 1

            # Set position / orientation of perturbation-visualisation capsule.
            # quat = np.zeros(4)
            # mat = np.zeros(9)
            # pertnorm = force / np.linalg.norm(force)
            # mj.mju_quatZ2Vec(quat, pertnorm)
            # mj.mju_quat2Mat(mat, quat)
            # self.data.geom_xpos[20] = ppos
            # self.data.geom_xmat[20] = mat

def main():
    xml_path = "humanoid/humanoid.xml"
    sim = Biped(xml_path)
    # sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
