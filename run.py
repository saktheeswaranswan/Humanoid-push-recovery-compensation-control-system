import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import scipy.linalg
from mujoco_base import MuJoCoBase

FSM_LEG1_SWING = 0
FSM_LEG2_SWING = 1

FSM_KNEE1_STANCE = 0
FSM_KNEE1_RETRACT = 1

FSM_KNEE2_STANCE = 0
FSM_KNEE2_RETRACT = 1


class Biped(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.simend = 30.0

        self.fsm_hip = None
        self.fsm_knee1 = None
        self.fsm_knee2 = None

    def reset(self, model, data):
        # Set camera configuration
        self.cam.azimuth = 120.89  # 89.608063
        self.cam.elevation = -15.81  # -11.588379
        self.cam.distance = 8.0  # 5.0
        self.cam.lookat = np.array([0.0, 0.0, 2.0])

        # self.data.ctrl = np.random.randn(self.model.nu)

        # self.model.opt.gravity[2] = -9.81

        mj.mj_resetDataKeyframe(model, data, 1)
        mj.mj_forward(model, data)
        data.qacc = 0  # Assert that there is no the acceleration.
        mj.mj_inverse(model, data)
        print(data.qfrc_inverse)

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

        # Plot the relationship.
        plt.figure(figsize=(10, 6))
        plt.plot(height_offsets * 1000, vertical_forces, linewidth=3)
        # Red vertical line at offset corresponding to smallest vertical force.
        plt.axvline(x=best_offset*1000, color='red', linestyle='--')
        # Green horizontal line at the humanoid's weight.
        weight = model.body_subtreemass[1]*np.linalg.norm(model.opt.gravity)
        plt.axhline(y=weight, color='green', linestyle='--')
        plt.xlabel('Height offset (mm)')
        plt.ylabel('Vertical force (N)')
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.title(f'Smallest vertical force '
                f'found at offset {best_offset*1000:.4f}mm.')
        # plt.show()

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

        DURATION = 12         # seconds
        FRAMERATE = 60        # Hz
        TOTAL_ROTATION = 15   # degrees
        CTRL_STD = 0.05       # actuator units
        CTRL_RATE = 0.8       # seconds

        # Define smooth orbiting function.
        def unit_smooth(normalised_time: float) -> float:
            return 1 - np.cos(normalised_time*2*np.pi)
        def azimuth(time: float) -> float:
            return 100 + unit_smooth(data.time/DURATION) * TOTAL_ROTATION

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

        return dq, qpos0, ctrl0, K, CTRL_STD, perturb

    def controller(self, model, data, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step):
        """
        This function implements a controller that 
        mimics the forces of a fixed joint before release
        """

        # Get state difference dx.
        mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T

        # LQR control law.
        data.ctrl = ctrl0 - K @ dx

        # Add perturbation, increment step.
        data.ctrl += CTRL_STD*perturb[step]

        

    def simulate(self):
        step = 0
        dq, qpos0, ctrl0, K, CTRL_STD, perturb = \
        self.reset(self.model, self.data)
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            # print(2)

            if self.data.time >= 10.0 and self.data.time <= 10.1  or self.data.time >= 20.0 and self.data.time <= 20.1:
                self.apply_external_forces([0.0, 5, 0.0])
            
            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)
                # print(1)

                # Apply control
                
                self.controller(self.model, self.data, dq, qpos0, ctrl0, K, CTRL_STD, perturb, step)
                step +=1
                if step >=2390:
                    step = 0

            if self.data.time >= self.simend:
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

    def apply_external_forces(self,force):

        self.data.qfrc_applied[0] = force[0]
        self.data.qfrc_applied[1] = force[1]
        self.data.qfrc_applied[2] = force[2]

        if np.linalg.norm(np.array(force)) != 0:
            print("Force is applied:",force[1]) 
            # Position and size of perturbation-visualisation capsule.
            bpos = self.data.xipos[1]
            ppos = bpos + force
            self.model.geom_size[2][1] = np.linalg.norm(force)

            # Make visualisation capsule visible.
            self.model.geom_rgba[2][3] = 1

            # Change colour of perturbed body.
            geoms = self.model.geom_bodyid
            self.model.geom_rgba[geoms,1] = 0.5

            # Set position / orientation of perturbation-visualisation capsule.
            quat = np.zeros(4)
            mat = np.zeros(9)
            pertnorm = force / np.linalg.norm(force)
            mj.mju_quatZ2Vec(quat, pertnorm)
            mj.mju_quat2Mat(mat, quat)
            self.data.geom_xpos[2] = ppos
            self.data.geom_xmat[2] = mat

def main():
    xml_path = "humanoid/humanoid.xml"
    sim = Biped(xml_path)
    # sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
