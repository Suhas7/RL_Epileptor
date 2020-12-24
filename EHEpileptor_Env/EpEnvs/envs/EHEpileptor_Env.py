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
        x1p = self.xhat_1()
        y1p = self.yhat_1()
        x2p = self.xhat_2()
        y2p = self.yhat_2()
        zp  = self.zhat()
        self.x1 += x1p * self.params['tstep']
        self.x2 += x2p * self.params['tstep']
        self.y1 += y1p * self.params['tstep']
        self.y2 += y2p * self.params['tstep']
        self.z  += zp  * self.params['tstep']

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
            lambda x: np.exp(-self.params['gamma'] * (self.frame - x)),
            0, self.frame
        )[0]

    def xhat_1(self) -> float:
        return self.y1 - self.f1() - self.z + self.params['I_ext1']

    def yhat_1(self) -> float:
        return self.params['c_1'] - self.params['d_1'] * (self.x1 ** 2) - self.y1

    def xhat_2(self) -> float:
        return -self.y2 + self.x2 - (self.x2 ** 3) + self.params['I_ext2'] \
               + 2 * self.g() - .3 * (self.z - 3.5)

    def yhat_2(self) -> float:
        return (-self.y2 + self.f2()) / self.params['tau_2']

    def zhat(self) -> float:
        if self.z < 0:
            return self.params['r'] * (self.params['s'] * (self.x1 - self.params['x0'])
                                       - self.z - .1 * (self.z ** 7))
        else:
            return self.params['r'] * (self.params['s'] * (self.x1 - self.params['x0'])
                                       - self.z)

    def __detect_SeizureState(self):
        pass

class JEpileptorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, params):
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
        self.params = params
        self.x1 = 0
        self.y1 = -5
        self.z = 3
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
        x1p = self.xhat_1() + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[0]
        y1p = self.yhat_1() + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[1]
        zp  = self.zhat()   + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[2]
        x2p = self.xhat_2() + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[3]
        y2p = self.yhat_2() + np.random.normal(loc = 0.0, scale = np.sqrt(self.params['tstep']))*sigmaNoise[4]
        self.x1 += x1p * self.params['tstep']
        self.x2 += x2p * self.params['tstep']
        self.y1 += y1p * self.params['tstep']
        self.y2 += y2p * self.params['tstep']
        self.z  += zp  * self.params['tstep']

    def f1(self) -> float:
        x1 = self.x1
        x2 = self.x2
        z = self.z
        if x1 < 0:
            return (x1 ** 3) - 3 * (x1 ** 2)
        else:
            return (x2 - .6 * ((z - 4) ** 2)) * x1

    def f2(self) -> float:
        x2 = self.x2
        if x2 >= -.25:
            return 6 * (x2 + .25)
        else:
            return 0

    def g(self) -> float:
        return self.x1*integrate.quad(
            lambda x: x*np.exp(-self.params['gamma'] * (self.frame - x)),
            0, self.frame
        )[0]

    def xhat_1(self) -> float:
        return self.y1 - self.f1() - self.z + self.params['I_rst1']

    def yhat_1(self) -> float:
        return self.params['y0'] - 5 * (self.x1 ** 2) - self.y1

    def zhat(self) -> float:
        return (4*(self.x1-self.params['x0'])-self.z)/self.params['tau0']

    def xhat_2(self) -> float:
        return -self.y2 + self.x2 - (self.x2 ** 3) + self.params['I_rst2'] \
               + .002 * self.g() - .3 * (self.z - 3.5)

    def yhat_2(self) -> float:
        return (-self.y2 + self.f2()) / self.params['tau2']

    def __detect_SeizureState(self) -> float:
        pass
