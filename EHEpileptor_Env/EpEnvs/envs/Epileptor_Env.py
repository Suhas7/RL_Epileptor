import gym
from gym import error, spaces, utils
import numpy as np
from EpEnvs.envs.Backends import JSim, EHSim
import matplotlib.pyplot as plt

class EpileptorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        """
        Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
        specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
        """
        self.config = config
        self.observation_space = spaces.Box(
            low=np.array([-15, -15, -15, -15, -15]),
            high=np.array([15, 15, 15, 15, 15])
        )
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([15, 5])
        )
        if config["backend"] == "EH":
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
            }
            self.sim = JSim()
        self.x1 = 0
        self.y1 = -5
        self.z = 5.5
        self.x2 = 0
        self.y2 = 0
        self.frame = 0
        self.curr_stim = []
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
        if len(self.curr_stim) == 0:
            self.curr_stim += [action[0]] * action[1] + [-action[0]] * action[1]
        self.system_step()
        self.frame += 1 / self.config["Fs"]
        self.history.append(self.x2-self.x1)
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
        plt.figure(figsize=(20,12))
        plt.plot(self.history)
        plt.show()

    def system_step(self):
        sigmaNoise = np.array([0.025, 0.025, 0.0, 0.25, 0.25, 0.]) * .1
        #noise = lambda p: np.random.normal(loc=0.0, scale=np.sqrt(1/self.config['Fs'])) * sigmaNoise[p]
        noise = lambda p: 0
        x1p = self.sim.xhat_1(self)
        y1p = self.sim.yhat_1(self)
        zp = self.sim.zhat(self)
        x2p = self.sim.xhat_2(self)
        y2p = self.sim.yhat_2(self)
        stim = 0
        if len(self.curr_stim)>0:
            stim=self.curr_stim[0]
            self.curr_stim=self.curr_stim[1:]
        self.x1 += x1p / self.config['Fs'] + noise(0) + .01*stim
        self.y1 += y1p / self.config['Fs'] + noise(1)
        self.z  += zp  / self.config['Fs'] + noise(2)
        self.x2 += x2p / self.config['Fs'] + noise(3) + .01*stim
        self.y2 += y2p / self.config['Fs'] + noise(4)
