import gym
from gym import error, spaces, utils
import numpy as np
from EpEnvs.envs.Backends import JSim, EHSim
Fs=512
finalTime = 100

class EpileptorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, backend):
        """
        Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
        specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
        """
        self.observation_space = spaces.Box(
            low=np.array([-999, -999, -999, -999, -999]),
            high=np.array([999, 999, 999, 999, 999])
        )
        self.action_space = spaces.Box(
            low=np.array([-999, -999, -999, -999, -999]),
            high=np.array([999, 999, 999, 999, 999])
        )
        if backend=="EH":
            self.params = {
                "a_1": 1,
                "b_1": 3,
                "c_1": 1,
                "d_1": 5,
                "I_ext1": 3.1,
                "m": 0,
                "a_2": 6,
                "tau_2": 10,
                "I_ext2": .45,
                "gamma": .01,
                "r": .00035,
                "s": 4,
                "x0": -1.6,
                "tstep": 1/Fs
            }
            self.sim = EHSim()
        else:
            self.params = {
                "x0": -1.6,
                "y0": 1,
                "tau0": 2857,
                "tau1": 1,
                "tau2": 10,
                "I_rst1": 3.1,
                "I_rst2": .45,
                "gamma": .01,
                "tstep": 1 / Fs
            }
            self.sim = JSim()
        self.x1 = 0
        self.y1 = -5
        self.z = 5.5
        self.x2 = 0
        self.y2 = 0
        self.frame = 0
        self.reset()
        self.history = list()

    def step(self, action) -> (list, float, bool):
        """
        This method is the primary interface between environment and agent.
        Parameters:
            action: int
        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        self.system_step()
        self.frame += self.params["tstep"]
        self.history.append(self.get_state())
        # todo process action
        return self.get_state(), 0, False

    def reset(self) -> list:
        """
        This method resets the environment to its initial values.
        Returns:
            observation:    array
                            the initial state of the environment
        """
        self.x1 = 0
        self.y1 = -1
        self.z = 3
        self.x2 = 0
        self.y2 = 0
        self.frame = 0
        return self.get_state()

    def get_state(self):
        return [self.x1, self.y1, self.z, self.x2, self.y2]

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        print(self.history)

    def system_step(self):
        sigmaNoise = np.array([0.025, 0.025, 0.0, 0.25, 0.25, 0.]) * 0.01
        x1p = self.sim.xhat_1(self) + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[0]
        y1p = self.sim.yhat_1(self) + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[1]
        zp  = self.sim.zhat(self)   + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[2]
        x2p = self.sim.xhat_2(self) + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[3]
        y2p = self.sim.yhat_2(self) + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[4]
        self.x1 += x1p * self.params['tstep']
        self.x2 += x2p * self.params['tstep']
        self.y1 += y1p * self.params['tstep']
        self.y2 += y2p * self.params['tstep']
        self.z  += zp  * self.params['tstep']

