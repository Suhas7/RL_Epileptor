from scipy import integrate
import numpy as np
class EHSim:
    def __init__(self):
        pass
    def f1(a,self) -> float:
        x1 = self.x1
        x2 = self.x2
        z = self.z
        if x1 < 0:
            return self.params['a_1'] * (x1 ** 3) - self.params['b_1'] * (x1 ** 2)
        else:
            return -(self.params['m'] - x2 + .6 * ((z - 4) ** 2)) * x1

    def f2(a,self) -> float:
        x2 = self.x2
        if x2 >= -.25:
            return self.params['a_2'] * (x2 + .25)
        else:
            return 0

    def g(a,self) -> float:
        return integrate.quad(
            lambda x: np.exp(-self.params['gamma'] * (self.frame - x)),
            0, self.frame
        )[0]

    def xhat_1(a,self) -> float:
        return self.y1 - a.f1(self) - self.z + self.params['I_ext1']

    def yhat_1(a,self) -> float:
        return self.params['c_1'] - self.params['d_1'] * (self.x1 ** 2) - self.y1

    def xhat_2(a,self) -> float:
        return -self.y2 + self.x2 - (self.x2 ** 3) + self.params['I_ext2'] \
               + 2 * a.g(self) - .3 * (self.z - 3.5)

    def yhat_2(a,self) -> float:
        return (-self.y2 + a.f2(self)) / self.params['tau_2']

    def zhat(a,self) -> float:
        if self.z < 0:
            return self.params['r'] * (self.params['s'] * (self.x1 - self.params['x0'])
                                       - self.z - .1 * (self.z ** 7))
        else:
            return self.params['r'] * (self.params['s'] * (self.x1 - self.params['x0'])
                                       - self.z)

    def __detect_SeizureState(self):
        pass

class JSim:
    def __init__(self):
        self.g_val=0

    def f1(a,self) -> float:
        x1 = self.x1
        x2 = self.x2
        z = self.z
        if x1 < 0:
            return (x1 ** 3) - 3 * (x1 ** 2)
        else:
            return (x2 - .6 * ((z - 4) ** 2)) * x1

    def f2(a,self) -> float:
        x2 = self.x2
        if x2 >= -.25:
            return 6 * (x2 + .25)
        else:
            return 0

    def g(a,self) -> float:
        return a.g_val
        return self.x1*integrate.quad(
            lambda x: x*np.exp(-self.params['gamma'] * (self.frame - x)),
            0, self.frame
        )[0]

    def xhat_1(a,self) -> float:
        return self.y1 - a.f1(self) - self.z + self.params['I_rst1']

    def yhat_1(a,self) -> float:
        return self.params['y0'] - 5 * (self.x1 ** 2) - self.y1

    def zhat(a,self) -> float:
        return (4*(self.x1-self.params['x0'])-self.z)/self.params['tau0']

    def xhat_2(a,self) -> float:
        return -self.y2 + self.x2 - (self.x2 ** 3) + self.params['I_rst2'] \
               + .002 * a.g_val - .3 * (self.z - 3.5)

    def yhat_2(a,self) -> float:
        return (-self.y2 + a.f2(self)) / self.params['tau2']

    def ghat(a, self):
        a.g_val+= self.params["tstep"]*(-self.params["gamma"] * (a.g_val - 0.1 * self.x1))

    def __detect_SeizureState(a,self) -> float:
        pass
