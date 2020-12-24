from EpEnvs.envs.Epileptor_Env import EpileptorEnv

config = {
    "backend":"J",
    "Fs": 512,
    "finalTime": 100,
}

env = EpileptorEnv(config)

data = []
for i in range(config["finalTime"] * config["Fs"]):
    # print(i)
    env.step(None)[0]

env.render()
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
