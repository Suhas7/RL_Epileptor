import gym
from gym import error, spaces, utils
from gym.utils import seeding
import scipy.integrate as integrate
import scipy.special as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class EHEpileptorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, params):
        """
        Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
        specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
        Example:
        >>> EnvTest = EpEnvs()
        >>> EnvTest.observation_space=spaces.Box(5)
        >>> EnvTest.action_space=spaces.Discrete(2)
        """
        self.observation_space = spaces.Box(
            low=np.array([-999, -999, -999, -999, -999]),
            high=np.array([999, 999, 999, 999, 999])
        )
        self.action_space = spaces.Box(
            low=np.array([-999, -999, -999, -999, -999]),
            high=np.array([999, 999, 999, 999, 999])
        )
        self.params = params
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.z = 0
        self.frame = 0
        self.reset()
        self.history = list()

    def step(self, action) -> (list, float, bool):
        """
        This method is the primary interface between environment and agent.
        Parameters:
            action: int
                    the index of the respective action (if action space is discrete)
        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        self.frame += self.params["tstep"]
        self.system_step()
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
        self.x2 = 0
        self.y1 = -5
        self.y2 = 0
        self.z = 3
        self.frame = 0
        return self.get_state()

    def get_state(self):
        return [self.x1, self.x2, self.y1, self.y2, self.z]

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        print(self.history)

    def system_step(self):
        x1 = self.__xhat_1()
        y1 = self.__yhat_1()
        x2 = self.__xhat_2()
        y2 = self.__yhat_2()
        z = self.__zhat()
        self.x1 = x1 * self.params['tstep'] + self.x1*(1-self.params['tstep'])
        self.x2 = x2 * self.params['tstep'] + self.x2*(1-self.params['tstep'])
        self.y1 = y1 * self.params['tstep'] + self.y1*(1-self.params['tstep'])
        self.y2 = y2 * self.params['tstep'] + self.y2*(1-self.params['tstep'])
        self.z  = z  * self.params['tstep'] + self.z*(1-self.params['tstep'])

    def f1(self) -> float:
        x1 = self.x1
        x2 = self.x2
        z = self.z
        if x1 < 0:
            return self.params['a_1'] * (x1 ** 3) - self.params['b_1'] * (x1 ** 2)
        else:
            return -(self.params['m'] - x2 + .6 * ((z - 4) ** 2)) * x1

    def f2(self) -> float:
        x2 = self.x2
        if x2 >= -.25:
            return self.params['a_2'] * (x2 + .25)
        else:
            return 0

    def g(self) -> float:
        return integrate.quad(
            lambda x, a, b: np.exp(-a * (b - x)),
            0, self.frame,
            args=(self.params['gamma'], self.frame)
        )[0]

    def __xhat_1(self) -> float:
        return self.y1 - self.f1() - self.z + self.params['I_ext1']

    def __yhat_1(self) -> float:
        return self.params['c_1'] - self.params['d_1'] * (self.x1 ** 2) - self.y1

    def __xhat_2(self) -> float:
        return -self.y2 + self.x2 - (self.x2 ** 3) + self.params['I_ext2'] \
               + 2 * self.g() - .3 * (self.z - 3.5)

    def __yhat_2(self) -> float:
        return (-self.y2 + self.f2()) / self.params['tau_2']

    def __zhat(self) -> float:
        if self.z < 0:
            return self.params['r'] * (self.params['s'] * (self.x1 - self.params['x0'])
                                       - self.z - .1 * (self.z ** 7))
        else:
            return self.params['r'] * (self.params['s'] * (self.x1 - self.params['x0'])
                                       - self.z)

    def __detect_SeizureState(self):
        pass