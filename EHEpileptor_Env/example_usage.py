from EpEnvs.envs.EHEpileptor_Env import EHEpileptorEnv, JEpileptorEnv
import matplotlib.pyplot as plt
Fs=512
finalTime = 1000

paramsEH = {
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

paramsJ = {
    "x0": -1.6,
    "y0": 1,
    "tau0": 2857,
    "tau1": 1,
    "tau2": 10,
    "I_rst1": 3.1,
    "I_rst2": .45,
    "gamma": .01,
    "tstep": 1/Fs
}

env = JEpileptorEnv(paramsJ)
#env = EHEpileptorEnv(paramsEH)
data=[]
for i in range(finalTime*Fs):
    #print(i)
    res=env.step(None)[0]
    data.append(res[3]-res[0])
plt.plot(data)
plt.show()

exit(0)
### Model Iteration

from stable_baselines import DDPG, PPO2
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np

n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

models = {
    "DDPG": DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise),
    "PPO2": PPO2(MlpPolicy, env, verbose=1)
}

for name, model in models.items():
    model.learn(total_timesteps=400000)
    model.save(name)
